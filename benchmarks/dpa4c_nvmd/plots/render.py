# SPDX-License-Identifier: LGPL-3.0-or-later
"""Matplotlib-backed plots with no import-time plotting dependency."""

from __future__ import (
    annotations,
)

import re
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Mapping,
    )


def _pyplot() -> Any:
    """Load matplotlib only when a user asks for figures."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plotting requires matplotlib; install it outside the core benchmark"
        ) from exc
    return plt


def _number(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Read a numeric result field from CSV or JSON records."""
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _save(figure: Any, output_dir: Path, name: str) -> Path:
    """Save one figure with a stable filename."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    figure.clf()
    return path


def _projection_reference(rows: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select the documented Air compressed baseline for the FPGA projection."""
    for row in rows:
        if row.get("model") == "Air" and row.get("path") == "compressed_generic":
            return row
    return max(
        rows,
        key=lambda row: (
            _number(row, "estimate_flops_total"),
            _number(row, "estimate_edges"),
        ),
    )


def plot_results(
    records: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    bandwidths_gbps: tuple[float, ...] = (25.0, 50.0, 100.0, 200.0),
    pe_counts: tuple[int, ...] = (16, 32, 64, 128),
    flops_per_pe: float = 2.0e9,
) -> list[Path]:
    """Generate the eight benchmark figures described by the benchmark plan."""
    if flops_per_pe <= 0.0:
        raise ValueError("flops_per_pe must be positive")
    rows = [row for row in records if row.get("status") in {"ok", "external"}]
    if not rows:
        raise ValueError("no successful benchmark records to plot")
    plt = _pyplot()
    output = Path(output_dir)
    paths: list[Path] = []

    figure, axis = plt.subplots()
    labels = [f"{row.get('model', '?')}/{row.get('path', '?')}" for row in rows]
    graph = [_number(row, "graph_time_ms") for row in rows]
    model = [_number(row, "step_time_ms") for row in rows]
    axis.bar(labels, graph, label="graph")
    axis.bar(labels, model, bottom=graph, label="model/fused")
    axis.set_ylabel("time (ms)")
    axis.set_title("Whole MD step time breakdown")
    axis.tick_params(axis="x", labelrotation=70)
    axis.legend()
    paths.append(_save(figure, output, "01_whole_step_breakdown"))

    figure, axis = plt.subplots()
    intensity = [_number(row, "estimate_arithmetic_intensity") for row in rows]
    flops_per_second = [
        _number(row, "estimate_flops_total")
        / max(_number(row, "time_ms") * 1.0e-3, 1.0e-30)
        for row in rows
    ]
    axis.scatter(intensity, flops_per_second)
    axis.set_xlabel("FLOP/byte")
    axis.set_ylabel("FLOP/s")
    axis.set_title("DPA4C analytical roofline points")
    axis.set_xscale("log")
    axis.set_yscale("log")
    paths.append(_save(figure, output, "02_roofline"))

    figure, axis = plt.subplots()
    axis.plot(
        [_number(row, "atoms") for row in rows],
        [_number(row, "atoms_per_second") for row in rows],
        "o-",
    )
    axis.set_xlabel("atoms")
    axis.set_ylabel("atoms/s")
    axis.set_title("Throughput vs atoms")
    axis.set_xscale("log")
    paths.append(_save(figure, output, "03_throughput_vs_atoms"))

    figure, axis = plt.subplots()
    axis.plot(
        [_number(row, "neighbors") for row in rows],
        [_number(row, "gedge_per_second") for row in rows],
        "o-",
    )
    axis.set_xlabel("neighbors/atom")
    axis.set_ylabel("Gedge/s")
    axis.set_title("Throughput vs neighbors")
    paths.append(_save(figure, output, "04_throughput_vs_neighbors"))

    figure, axis = plt.subplots()
    axis.bar(
        [str(row.get("model", "?")) for row in rows],
        [_number(row, "time_ms") for row in rows],
    )
    axis.set_xlabel("(C0, L, R) model profile")
    axis.set_ylabel("time (ms)")
    axis.set_title("Runtime vs DPA4C architecture profile")
    paths.append(_save(figure, output, "05_runtime_vs_profile"))

    figure, axis = plt.subplots()
    axis.plot(
        [_number(row, "atoms") for row in rows],
        [_number(row, "estimate_total_step_bytes") / 2**20 for row in rows],
        "o-",
    )
    axis.set_xlabel("atoms")
    axis.set_ylabel("compulsory traffic (MiB)")
    axis.set_title("Memory traffic vs atoms/model")
    axis.set_xscale("log")
    paths.append(_save(figure, output, "06_memory_vs_atoms"))

    figure, axis = plt.subplots()
    bits = [
        _precision_bits(str(row.get("precision", "")))
        for row in rows
        if "force_rmse" in row
    ]
    force_error = [_number(row, "force_rmse") for row in rows if "force_rmse" in row]
    if bits:
        axis.plot(bits, force_error, "o-")
    axis.set_xlabel("precision bits")
    axis.set_ylabel("force RMSE")
    axis.set_title("Quantization error vs bit width")
    paths.append(_save(figure, output, "07_quantization_error"))

    figure, axis = plt.subplots()
    reference = _projection_reference(rows)
    flops = _number(reference, "estimate_flops_total")
    traffic = _number(reference, "estimate_total_step_bytes")
    for bandwidth in bandwidths_gbps:
        throughput = []
        for pe_count in pe_counts:
            compute = pe_count * flops_per_pe
            memory = bandwidth * 1.0e9 / max(traffic, 1.0)
            throughput.append(min(compute / max(flops, 1.0), memory))
        axis.plot(pe_counts, throughput, label=f"{bandwidth:g} GB/s")
    axis.set_xlabel("PE count")
    axis.set_ylabel("projected steps/s")
    axis.set_title("Projected FPGA speedup design space")
    axis.legend()
    paths.append(_save(figure, output, "08_fpga_projection"))
    return paths


def _precision_bits(value: str) -> float:
    """Extract a displayable bit count from a precision label."""
    match = re.search(r"(\d+)", value)
    return 0.0 if match is None else float(match.group(1))
