# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dependency-light mixed-precision experiments for DPA4C outputs."""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Mapping,
    )

import numpy as np


@dataclass(frozen=True)
class QuantizationSpec:
    """One software-emulated arithmetic format."""

    bits: int
    kind: str = "symmetric"
    name: str = ""

    def label(self) -> str:
        """Return a stable display label."""
        return self.name or f"{self.kind}{self.bits}"


def quantize(
    values: Any,
    spec: QuantizationSpec,
) -> tuple[np.ndarray, float]:
    """Quantize an array and return values plus the applied scale."""
    array = np.asarray(values)
    if spec.bits <= 0:
        raise ValueError("quantization bits must be positive")
    if spec.kind == "bfloat16":
        # bfloat16 has seven explicit mantissa bits. Operate on the
        # significand so the emulator remains available without torch.
        return _quantize_mantissa(array, 7), 1.0
    if spec.kind in {"float32", "float16"}:
        dtype = {
            "float32": np.float32,
            "float16": np.float16,
        }[spec.kind]
        result = array.astype(dtype).astype(np.float64)
        return result, 1.0
    if spec.kind in {"float24", "fixed", "symmetric"}:
        if spec.kind == "float24":
            return _quantize_mantissa(array, max(2, spec.bits - 9)), 1.0
        peak = float(np.max(np.abs(array))) if array.size else 0.0
        if peak == 0.0:
            return np.zeros_like(array, dtype=np.float64), 1.0
        levels = 2 ** (spec.bits - 1) - 1
        scale = peak / levels
        return np.round(array / scale).clip(-levels, levels) * scale, scale
    if spec.kind == "int":
        peak = float(np.max(np.abs(array))) if array.size else 0.0
        levels = 2 ** (spec.bits - 1) - 1
        scale = 1.0 if peak == 0.0 else peak / levels
        return (
            np.round(array / scale).clip(-levels, levels) * scale,
            scale,
        )
    raise ValueError(f"unsupported quantization kind {spec.kind}")


def _quantize_mantissa(values: Any, mantissa_bits: int) -> np.ndarray:
    """Round normalized mantissas while retaining the exponent."""
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return array.copy()
    mantissa, exponent = np.frexp(array)
    quantum = 2.0 ** (-mantissa_bits)
    rounded = np.round(mantissa / quantum) * quantum
    return np.ldexp(rounded, exponent)


def quantize_lut(
    function: Callable[[np.ndarray], Any],
    *,
    rcut: float,
    spacing: float,
    spec: QuantizationSpec,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample, quantize, and return a uniform radial lookup table."""
    if rcut <= 0.0 or spacing <= 0.0:
        raise ValueError("rcut and spacing must be positive")
    count = int(np.ceil(rcut / spacing))
    grid = np.arange(count + 1, dtype=np.float64) * spacing
    grid[-1] = rcut
    values, scale = quantize(function(grid), spec)
    return grid, values, scale


def default_lut_sweep() -> tuple[tuple[float, QuantizationSpec], ...]:
    """Return the first radial spacing/bit-width experiment matrix."""
    return tuple(
        (spacing, QuantizationSpec(bits, kind="fixed", name=f"fixed{bits}"))
        for spacing in (0.001, 0.002, 0.005, 0.01)
        for bits in (32, 24, 16, 12)
    )


def compare_observables(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, float]:
    """Compare E/F/V arrays with MAE, RMSE, and worst-case force errors."""
    result: dict[str, float] = {}
    if "energy" in reference and "energy" in candidate:
        result["energy_mae"] = _mae(reference["energy"], candidate["energy"])
    if "force" in reference and "force" in candidate:
        difference = np.asarray(candidate["force"]) - np.asarray(reference["force"])
        result["force_rmse"] = float(np.sqrt(np.mean(difference**2)))
        result["force_max"] = float(np.max(np.abs(difference)))
    if "virial" in reference and "virial" in candidate:
        result["virial_error"] = _mae(reference["virial"], candidate["virial"])
        result["virial_max"] = float(
            np.max(np.abs(np.asarray(candidate["virial"]) - reference["virial"]))
        )
    return result


def _mae(reference: Any, candidate: Any) -> float:
    """Return mean absolute error in float64."""
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(
        reference,
        dtype=np.float64,
    )
    return float(np.mean(np.abs(difference)))


def symmetry_metrics(
    energy: float,
    force: Any,
    transformed_energy: float,
    transformed_force: Any,
    rotation: Any,
) -> dict[str, float]:
    """Check scalar invariance and vector equivariance for one transform."""
    matrix = np.asarray(rotation, dtype=np.float64)
    force_array = np.asarray(force, dtype=np.float64)
    expected = force_array @ matrix.T
    transformed = np.asarray(transformed_force, dtype=np.float64)
    return {
        "symmetry_energy_error": float(abs(float(transformed_energy) - energy)),
        "force_equivariance_max": float(np.max(np.abs(transformed - expected))),
    }


def finite_difference_force(
    energy_function: Callable[[np.ndarray], float],
    coordinates: Any,
    *,
    delta: float = 1.0e-4,
) -> np.ndarray:
    """Estimate ``-dE/dr`` by a central finite difference."""
    if delta <= 0.0:
        raise ValueError("finite-difference delta must be positive")
    coordinate = np.asarray(coordinates, dtype=np.float64).copy()
    gradient = np.empty_like(coordinate)
    for index in np.ndindex(coordinate.shape):
        plus = coordinate.copy()
        minus = coordinate.copy()
        plus[index] += delta
        minus[index] -= delta
        gradient[index] = -(
            float(energy_function(plus)) - float(energy_function(minus))
        ) / (2.0 * delta)
    return gradient


def md_validation_metrics(
    *,
    nve_energy: Any | None = None,
    nvt_temperature: Any | None = None,
    npt_pressure: Any | None = None,
    npt_volume: Any | None = None,
) -> dict[str, float]:
    """Summarize trajectory stability checks for E/F/V candidates."""
    result: dict[str, float] = {}
    if nve_energy is not None:
        energy = np.asarray(nve_energy, dtype=np.float64)
        result["nve_drift"] = float(
            (energy[-1] - energy[0]) / max(1, energy.shape[0] - 1)
        )
        result["nve_range"] = float(np.max(energy) - np.min(energy))
    if nvt_temperature is not None:
        temperature = np.asarray(nvt_temperature, dtype=np.float64)
        result["nvt_temperature_mean"] = float(np.mean(temperature))
        result["nvt_temperature_std"] = float(np.std(temperature))
    if npt_pressure is not None:
        pressure = np.asarray(npt_pressure, dtype=np.float64)
        result["npt_pressure_mean"] = float(np.mean(pressure))
        result["npt_pressure_std"] = float(np.std(pressure))
    if npt_volume is not None:
        volume = np.asarray(npt_volume, dtype=np.float64)
        result["npt_volume_mean"] = float(np.mean(volume))
        result["npt_volume_std"] = float(np.std(volume))
    return result
