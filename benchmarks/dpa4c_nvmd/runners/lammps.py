# SPDX-License-Identifier: LGPL-3.0-or-later
"""External LAMMPS/Kokkos execution adapter for benchmark cases."""

from __future__ import (
    annotations,
)

import re
import shlex
import subprocess
import time
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from collections.abc import (
        Mapping,
        Sequence,
    )
    from pathlib import (
        Path,
    )

    from ..contract import (
        BenchmarkCase,
    )

from ..metrics import (
    estimate_workload,
)
from ..results import (
    EnvironmentMetadata,
    result_row,
)

_LOOP_TIME = re.compile(r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs")


def run_lammps_case(
    case: BenchmarkCase,
    input_file: str | Path,
    *,
    command: str | Sequence[str] = "lmp",
    cwd: str | Path | None = None,
    variables: Mapping[str, str | int | float] | None = None,
    timeout: float | None = None,
    environment: EnvironmentMetadata | None = None,
) -> dict[str, Any]:
    """Run one input without a shell and normalize its reported loop time."""
    case.validate()
    if case.path != "lammps":
        raise ValueError("the LAMMPS adapter requires path='lammps'")
    executable = shlex.split(command) if isinstance(command, str) else list(command)
    if not executable:
        raise ValueError("LAMMPS command must not be empty")
    args = [*executable, "-in", str(input_file)]
    for name, value in (variables or {}).items():
        args.extend(["-var", str(name), str(value)])
    estimate = estimate_workload(case)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return result_row(
            case,
            estimate,
            environment=environment,
            status="failed",
            error=str(exc),
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    output = f"{completed.stdout}\n{completed.stderr}"
    match = _LOOP_TIME.search(output)
    timing: dict[str, Any] = {
        "wall_time_ms": elapsed_ms,
        "time_ms": elapsed_ms,
    }
    if match is not None:
        seconds = float(match.group(1))
        timing.update(
            {
                "lammps_loop_time_s": seconds,
                "lammps_processes": int(match.group(2)),
                "time_ms": seconds * 1000.0,
                "edges_per_second": case.edges / seconds,
                "atoms_per_second": case.atoms / seconds,
                "gedge_per_second": case.edges / seconds / 1.0e9,
            }
        )
    return result_row(
        case,
        estimate,
        environment=environment,
        status="ok" if completed.returncode == 0 else "failed",
        error="" if completed.returncode == 0 else f"returncode={completed.returncode}",
        timing=timing,
    )
