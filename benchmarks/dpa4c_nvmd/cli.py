# SPDX-License-Identifier: LGPL-3.0-or-later
"""Command-line entry point for the DPA4C NvNMD benchmark scaffold."""

from __future__ import (
    annotations,
)

import argparse
import sys
from pathlib import (
    Path,
)
from typing import (
    Any,
)

from .contract import (
    iter_cases,
    load_matrix,
)
from .metrics import (
    estimate_workload,
)
from .plots import (
    plot_results,
)
from .results import (
    collect_environment,
    read_results,
    result_row,
    write_results,
)
from .runners.lammps import (
    run_lammps_case,
)
from .runners.synthetic import (
    run_synthetic_case,
)

_ROOT = Path(__file__).resolve().parent
_DEFAULT_MATRIX = _ROOT / "configs" / "matrix.json"


def _common_parser(parser: argparse.ArgumentParser) -> None:
    """Add matrix and case-selection flags shared by estimate and run."""
    parser.add_argument(
        "--matrix",
        type=Path,
        default=_DEFAULT_MATRIX,
        help="benchmark matrix JSON",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="model name; repeat to select multiple models",
    )
    parser.add_argument(
        "--path",
        action="append",
        dest="paths",
        help="execution path; repeat to select multiple paths",
    )
    parser.add_argument(
        "--sweep",
        choices=(
            "point",
            "all",
            "atoms",
            "neighbors",
            "active_types",
            "modes",
            "tiles",
            "table_spacings",
        ),
        default="point",
    )
    parser.add_argument(
        "--precision",
        default="float32",
        help="contract precision label, including emulator-only formats",
    )


def _cases(arguments: argparse.Namespace) -> tuple[dict[str, Any], list[Any]]:
    """Load a matrix and construct its non-Cartesian case list."""
    matrix = load_matrix(arguments.matrix)
    cases = iter_cases(
        matrix,
        models=("Air",) if arguments.models is None else tuple(arguments.models),
        paths=None if arguments.paths is None else tuple(arguments.paths),
        sweep=arguments.sweep,
        precision=arguments.precision,
    )
    if getattr(arguments, "max_cases", None) is not None:
        cases = cases[: arguments.max_cases]
    return matrix, cases


def _estimate(arguments: argparse.Namespace) -> int:
    """Write analytical workload estimates without importing torch."""
    _matrix, cases = _cases(arguments)
    environment = collect_environment()
    rows = [
        result_row(
            case,
            estimate_workload(case),
            environment=environment,
            status="estimate",
        )
        for case in cases
    ]
    write_results(rows, arguments.output, arguments.json_output)
    return 0


def _run(arguments: argparse.Namespace) -> int:
    """Run synthetic or external LAMMPS cases."""
    _matrix, cases = _cases(arguments)
    environment = collect_environment(model_path=arguments.model_file)
    rows = []
    for case in cases:
        if arguments.dry_run:
            rows.append(
                result_row(
                    case,
                    estimate_workload(case),
                    environment=environment,
                    status="estimate",
                )
            )
        elif case.path == "lammps":
            if arguments.lammps_input is None:
                rows.append(
                    result_row(
                        case,
                        estimate_workload(case),
                        environment=environment,
                        status="not_run",
                        error="--lammps-input is required for path=lammps",
                    )
                )
            else:
                rows.append(
                    run_lammps_case(
                        case,
                        arguments.lammps_input,
                        command=arguments.lammps_command,
                        timeout=arguments.timeout,
                        environment=environment,
                    )
                )
        else:
            rows.append(
                run_synthetic_case(
                    case,
                    device=arguments.device,
                    seed=arguments.seed,
                    warmup=arguments.warmup,
                    repetitions=arguments.repetitions,
                    environment=environment,
                )
            )
    write_results(rows, arguments.output, arguments.json_output)
    return 0


def _plot(arguments: argparse.Namespace) -> int:
    """Generate the eight figures from a CSV or JSON result file."""
    rows = read_results(arguments.input)
    paths = plot_results(rows, arguments.output_dir)
    sys.stdout.write("\n".join(str(path) for path in paths) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the lazy-import CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.dpa4c_nvmd",
        description="DPA4C-to-NvNMD pre-silicon benchmark scaffold",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser("estimate", help="write FLOP/byte estimates")
    _common_parser(estimate)
    estimate.add_argument("--output", type=Path, required=True)
    estimate.add_argument("--json-output", type=Path)
    estimate.set_defaults(function=_estimate)

    run = subparsers.add_parser("run", help="run synthetic or LAMMPS cases")
    _common_parser(run)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--json-output", type=Path)
    run.add_argument("--device", default="cpu")
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--warmup", type=int, default=1)
    run.add_argument("--repetitions", type=int, default=3)
    run.add_argument("--max-cases", type=int)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--model-file", type=Path)
    run.add_argument("--lammps-input", type=Path)
    run.add_argument("--lammps-command", default="lmp")
    run.add_argument("--timeout", type=float)
    run.set_defaults(function=_run)

    plot = subparsers.add_parser("plot", help="generate the eight result figures")
    plot.add_argument("--input", type=Path, required=True)
    plot.add_argument("--output-dir", type=Path, required=True)
    plot.set_defaults(function=_plot)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark command."""
    arguments = build_parser().parse_args(argv)
    return int(arguments.function(arguments))
