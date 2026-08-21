# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dependency-light phase timing with optional CUDA events and NVTX."""

from __future__ import (
    annotations,
)

import time
from contextlib import (
    contextmanager,
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
        Iterator,
        Mapping,
    )


@dataclass(frozen=True)
class PhaseTiming:
    """Summary of repeated phase measurements in milliseconds."""

    name: str
    time_ms: float
    min_ms: float
    max_ms: float
    repetitions: int
    launches: int = 1

    def as_dict(self) -> dict[str, Any]:
        """Return flat timing fields suitable for a result row."""
        return {
            f"{self.name}_time_ms": self.time_ms,
            f"{self.name}_min_ms": self.min_ms,
            f"{self.name}_max_ms": self.max_ms,
            f"{self.name}_repetitions": self.repetitions,
            f"{self.name}_launches": self.launches,
        }


def _torch_module() -> Any | None:
    """Import PyTorch only when timing actually requests it."""
    try:
        import torch
    except ImportError:
        return None
    return torch


@contextmanager
def nvtx_range(name: str, enabled: bool = True) -> Iterator[None]:
    """Annotate a logical phase when CUDA NVTX is available."""
    torch = _torch_module() if enabled else None
    active = bool(torch is not None and torch.cuda.is_available())
    if active:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if active:
            torch.cuda.nvtx.range_pop()


def _cuda_available(device: str | None) -> tuple[Any | None, bool]:
    """Return a lazy torch module and whether CUDA event timing is usable."""
    torch = _torch_module()
    if torch is None or not torch.cuda.is_available():
        return torch, False
    if device is not None and not str(device).startswith("cuda"):
        return torch, False
    return torch, True


def measure(
    name: str,
    function: Callable[[], Any],
    *,
    warmup: int = 1,
    repetitions: int = 5,
    device: str | None = None,
    launches: int = 1,
) -> tuple[PhaseTiming, Any]:
    """Measure a callable, using CUDA events when the device supports them."""
    if warmup < 0 or repetitions <= 0:
        raise ValueError("warmup must be non-negative and repetitions positive")
    torch, use_cuda = _cuda_available(device)
    with nvtx_range(f"benchmark:{name}"):
        for _ in range(warmup):
            function()
        if use_cuda and torch is not None:
            torch.cuda.synchronize()
            samples = []
            output = None
            for _ in range(repetitions):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                output = function()
                stop.record()
                stop.synchronize()
                samples.append(float(start.elapsed_time(stop)))
        else:
            samples = []
            output = None
            for _ in range(repetitions):
                start_time = time.perf_counter()
                output = function()
                samples.append((time.perf_counter() - start_time) * 1000.0)
    summary = PhaseTiming(
        name=name,
        time_ms=sum(samples) / len(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        repetitions=repetitions,
        launches=launches,
    )
    return summary, output


def profile_phases(
    phases: Mapping[str, Callable[[], Any]],
    *,
    warmup: int = 1,
    repetitions: int = 5,
    device: str | None = None,
) -> tuple[dict[str, PhaseTiming], dict[str, Any]]:
    """Measure named logical phases and return their last outputs."""
    timings: dict[str, PhaseTiming] = {}
    outputs: dict[str, Any] = {}
    for name, function in phases.items():
        timing, output = measure(
            name,
            function,
            warmup=warmup,
            repetitions=repetitions,
            device=device,
        )
        timings[name] = timing
        outputs[name] = output
    return timings, outputs
