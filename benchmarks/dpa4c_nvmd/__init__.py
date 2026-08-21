# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pre-silicon workload accounting and runners for DPA4C NvNMD."""

from .contract import (
    BenchmarkCase,
    ModelSpec,
    iter_cases,
    load_matrix,
)
from .metrics import (
    estimate_workload,
    profile_dimensions,
)
from .results import (
    collect_environment,
    read_results,
    write_results,
)

__all__ = [
    "BenchmarkCase",
    "ModelSpec",
    "collect_environment",
    "estimate_workload",
    "iter_cases",
    "load_matrix",
    "profile_dimensions",
    "read_results",
    "write_results",
]
