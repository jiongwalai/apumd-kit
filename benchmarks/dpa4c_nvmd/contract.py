# SPDX-License-Identifier: LGPL-3.0-or-later
"""Configuration and result contracts for the DPA4C NvNMD benchmark."""

from __future__ import (
    annotations,
)

import json
from dataclasses import (
    asdict,
    dataclass,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

SUPPORTED_CHANNELS = (8, 16, 32, 64, 128)
SUPPORTED_LMAX = (2, 3, 4)
SUPPORTED_RADIAL_MODES = (0, 2, 4, 8)
SUPPORTED_PATHS = (
    "uncompressed",
    "compressed_generic",
    "compressed_canonical",
    "lammps",
)
SUPPORTED_MODES = ("energy", "energy_force", "energy_force_virial")


@dataclass(frozen=True)
class ModelSpec:
    """Describe one compiled DPA4C grade."""

    name: str
    channels: int
    lmax: int
    radial_modes: int
    fitting_width: int

    def validate(self) -> None:
        """Validate the structural parameters accepted by the CUDA path."""
        if self.channels not in SUPPORTED_CHANNELS:
            raise ValueError(
                f"{self.name}: channels must be one of {SUPPORTED_CHANNELS}, "
                f"got {self.channels}"
            )
        if self.lmax not in SUPPORTED_LMAX:
            raise ValueError(
                f"{self.name}: lmax must be one of {SUPPORTED_LMAX}, got {self.lmax}"
            )
        if self.radial_modes not in SUPPORTED_RADIAL_MODES:
            raise ValueError(
                f"{self.name}: radial_modes must be one of "
                f"{SUPPORTED_RADIAL_MODES}, got {self.radial_modes}"
            )
        if self.fitting_width <= 0:
            raise ValueError(f"{self.name}: fitting_width must be positive")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkCase:
    """One point in the benchmark matrix."""

    model: ModelSpec
    path: str
    atoms: int
    neighbors: int
    active_types: int
    mode: str
    tile: int
    precision: str = "float32"
    table_spacing: float = 0.002

    @property
    def edges(self) -> int:
        """Return the directed edge count for this synthetic workload."""
        return self.atoms * self.neighbors

    @property
    def model_types(self) -> int:
        """Return the type-table size required by the compressed ABI.

        The current operator requires at least two rows. A one-species workload
        therefore uses a two-row model table while retaining one active species.
        """
        return max(2, self.active_types)

    def validate(self) -> None:
        """Validate one benchmark point."""
        self.model.validate()
        if self.path not in SUPPORTED_PATHS:
            raise ValueError(f"path must be one of {SUPPORTED_PATHS}, got {self.path}")
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {SUPPORTED_MODES}, got {self.mode}")
        if self.atoms <= 0 or self.neighbors <= 0:
            raise ValueError("atoms and neighbors must be positive")
        if self.active_types <= 0:
            raise ValueError("active_types must be positive")
        if self.tile <= 0:
            raise ValueError("tile must be positive")
        if self.precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"unsupported precision {self.precision}")
        if self.table_spacing <= 0.0:
            raise ValueError("table_spacing must be positive")

    def as_dict(self) -> dict[str, Any]:
        """Return a flattened JSON-compatible case description."""
        self.validate()
        result = asdict(self)
        result["model"] = self.model.as_dict()
        result["edges"] = self.edges
        result["model_types"] = self.model_types
        return result


DEFAULT_MODELS = {
    "Nano": ModelSpec("Nano", 8, 2, 0, 96),
    "Mini": ModelSpec("Mini", 32, 2, 0, 192),
    "Neo": ModelSpec("Neo", 32, 2, 4, 192),
    "Air": ModelSpec("Air", 64, 3, 4, 256),
    "Plus": ModelSpec("Plus", 128, 3, 4, 384),
}


def _parse_model(name: str, value: dict[str, Any]) -> ModelSpec:
    """Parse a model entry from the matrix JSON."""
    model = ModelSpec(
        name=name,
        channels=int(value["channels"]),
        lmax=int(value["lmax"]),
        radial_modes=int(value["radial_modes"]),
        fitting_width=int(value["fitting_width"]),
    )
    model.validate()
    return model


def load_matrix(path: str | Path) -> dict[str, Any]:
    """Load and validate a benchmark matrix JSON file."""
    matrix_path = Path(path)
    with matrix_path.open(encoding="utf-8") as handle:
        matrix = json.load(handle)
    if matrix.get("schema_version") != 1:
        raise ValueError("benchmark matrix schema_version must be 1")

    models = {
        name: _parse_model(name, value)
        for name, value in matrix.get("models", {}).items()
    }
    if not models:
        raise ValueError("benchmark matrix must define at least one model")
    sweeps = matrix.get("sweeps", {})
    required_sweeps = (
        "atoms",
        "neighbors",
        "active_types",
        "modes",
        "tiles",
        "table_spacings",
    )
    missing = [name for name in required_sweeps if name not in sweeps]
    if missing:
        raise ValueError(f"benchmark matrix is missing sweeps: {missing}")
    if any(int(value) <= 0 for value in sweeps["atoms"]):
        raise ValueError("all atom counts must be positive")
    if any(int(value) <= 0 for value in sweeps["neighbors"]):
        raise ValueError("all neighbor counts must be positive")
    if any(int(value) <= 0 for value in sweeps["active_types"]):
        raise ValueError("all active type counts must be positive")
    if any(mode not in SUPPORTED_MODES for mode in sweeps["modes"]):
        raise ValueError("matrix contains an unsupported mode")
    if any(int(value) <= 0 for value in sweeps["tiles"]):
        raise ValueError("all tile sizes must be positive")
    if any(float(value) <= 0.0 for value in sweeps["table_spacings"]):
        raise ValueError("all table spacings must be positive")

    paths = matrix.get("paths", list(SUPPORTED_PATHS))
    invalid_paths = [
        path_name for path_name in paths if path_name not in SUPPORTED_PATHS
    ]
    if invalid_paths:
        raise ValueError(f"matrix contains unsupported paths: {invalid_paths}")
    return {
        "schema_version": 1,
        "models": models,
        "sweeps": sweeps,
        "paths": paths,
        "notes": matrix.get("notes", []),
        "source": str(matrix_path),
    }
