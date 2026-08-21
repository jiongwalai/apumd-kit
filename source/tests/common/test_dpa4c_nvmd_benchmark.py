# SPDX-License-Identifier: LGPL-3.0-or-later
"""Focused contract tests for the DPA4C NvNMD benchmark scaffold."""

from pathlib import (
    Path,
)

import numpy as np

from benchmarks.dpa4c_nvmd.contract import (
    BenchmarkCase,
    iter_cases,
    load_matrix,
)
from benchmarks.dpa4c_nvmd.metrics import (
    estimate_workload,
    profile_dimensions,
)
from benchmarks.dpa4c_nvmd.precision import (
    QuantizationSpec,
    compare_observables,
    finite_difference_force,
    md_validation_metrics,
    quantize,
)
from benchmarks.dpa4c_nvmd.results import (
    read_results,
    result_row,
    write_results,
)

_ROOT = Path(__file__).resolve().parents[3]
_MATRIX = _ROOT / "benchmarks" / "dpa4c_nvmd" / "configs" / "matrix.json"


def test_profiles_match_compiled_dpa4c_widths() -> None:
    """Keep the analytical profile aligned with the compiled descriptor."""
    matrix = load_matrix(_MATRIX)
    expected = {
        "Nano": (40, 70),
        "Mini": (76, 144),
        "Neo": (76, 144),
        "Air": (115, 219),
        "Plus": (223, 541),
    }
    for name, (moment_width, output_width) in expected.items():
        profile = profile_dimensions(matrix["models"][name])
        assert (profile.moment_width, profile.output_width) == (
            moment_width,
            output_width,
        )


def test_one_active_type_keeps_two_row_model_table() -> None:
    """Document the compressed operator's type-count ABI workaround."""
    matrix = load_matrix(_MATRIX)
    case = iter_cases(
        matrix,
        models=("Air",),
        paths=("compressed_generic",),
        sweep="point",
    )[0]
    assert case.active_types == 1
    assert case.model_types == 2


def test_case_sweeps_are_not_cartesian() -> None:
    """The all-sweep plan varies one dimension at a time."""
    matrix = load_matrix(_MATRIX)
    cases = iter_cases(
        matrix,
        models=("Air",),
        paths=("compressed_generic",),
        sweep="all",
    )
    assert len(cases) == sum(len(values) for values in matrix["sweeps"].values())
    assert {case.neighbors for case in cases if case.atoms == 2000} >= {
        32,
        256,
    }


def test_workload_scales_edges_and_exposes_recompute_tradeoff() -> None:
    """Traffic and edge work scale linearly with the synthetic graph."""
    matrix = load_matrix(_MATRIX)
    model = matrix["models"]["Air"]
    small = BenchmarkCase(model, "compressed_generic", 100, 32, 1, "energy", 1024)
    large = BenchmarkCase(model, "compressed_generic", 200, 32, 1, "energy", 1024)
    small_estimate = estimate_workload(small)
    large_estimate = estimate_workload(large)
    assert large_estimate.edges == 2 * small_estimate.edges
    assert large_estimate.flops_total > small_estimate.flops_total
    assert small_estimate.save_step_bytes > small_estimate.recompute_step_bytes
    assert small_estimate.model_cache_bytes < small_estimate.model_bytes
    canonical = estimate_workload(
        BenchmarkCase(model, "compressed_canonical", 100, 32, 1, "energy", 1024)
    )
    assert canonical.edge_bytes < small_estimate.edge_bytes


def test_result_roundtrip_and_precision_metrics(tmp_path: Path) -> None:
    """Result rows and numerical hooks remain dependency-light."""
    matrix = load_matrix(_MATRIX)
    case = iter_cases(
        matrix,
        models=("Nano",),
        paths=("uncompressed",),
        sweep="point",
    )[0]
    estimate = estimate_workload(case)
    row = result_row(
        case,
        estimate,
        status="ok",
        numeric={"force_rmse": 0.1},
    )
    csv_path = tmp_path / "result.csv"
    json_path = tmp_path / "result.json"
    write_results([row], csv_path, json_path)
    assert read_results(csv_path)[0]["model"] == "Nano"
    assert read_results(json_path)[0]["force_rmse"] == 0.1

    quantized, scale = quantize(
        np.array([-1.0, 0.0, 0.5]),
        QuantizationSpec(8),
    )
    assert scale > 0.0
    assert np.max(np.abs(quantized - np.array([-1.0, 0.0, 0.5]))) < 0.01
    reference = {"energy": [1.0], "force": [[1.0, 0.0, 0.0]]}
    candidate = {"energy": [1.1], "force": [[1.2, 0.0, 0.0]]}
    errors = compare_observables(reference, candidate)
    np.testing.assert_allclose(errors["force_max"], 0.2)


def test_derivative_and_md_validation_hooks() -> None:
    """Finite-difference and trajectory summaries expose required diagnostics."""
    coordinates = np.array([[1.0, -2.0, 0.5]])
    force = finite_difference_force(
        lambda values: float(np.sum(values**2)),
        coordinates,
    )
    np.testing.assert_allclose(force, -2.0 * coordinates, atol=1.0e-6)
    metrics = md_validation_metrics(
        nve_energy=[0.0, 0.2, 0.4],
        nvt_temperature=[300.0, 302.0],
        npt_pressure=[1.0, 3.0],
        npt_volume=[10.0, 11.0],
    )
    assert metrics["nve_drift"] == 0.2
    assert metrics["nvt_temperature_mean"] == 301.0
