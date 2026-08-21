# SPDX-License-Identifier: LGPL-3.0-or-later
"""Small deterministic graph runner for pre-silicon DPA4C comparisons."""

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
    from ..contract import (
        BenchmarkCase,
    )

from ..instrumentation import (
    measure,
    nvtx_range,
)
from ..metrics import (
    estimate_workload,
)
from ..results import (
    EnvironmentMetadata,
    result_row,
)


@dataclass
class SyntheticGraph:
    """Torch-backed regular directed graph used for controlled sweeps."""

    graph: Any
    atype: Any

    @property
    def atoms(self) -> int:
        """Return the node count."""
        return int(self.atype.shape[0])

    @property
    def edges(self) -> int:
        """Return the physical edge count."""
        return int(self.graph.edge_index.shape[1])


def _require_torch() -> Any:
    """Import PyTorch only when a synthetic execution is requested."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "the synthetic runner requires PyTorch; use --dry-run for estimates"
        ) from exc
    return torch


def build_synthetic_graph(
    case: BenchmarkCase,
    *,
    device: str = "cpu",
    seed: int = 17,
) -> SyntheticGraph:
    """Build a regular graph with the requested atoms, degree, and type count."""
    case.validate()
    torch = _require_torch()
    generator = torch.Generator(device=device).manual_seed(seed)
    atoms = case.atoms
    edges = case.edges
    destination = torch.arange(
        atoms, device=device, dtype=torch.int64
    ).repeat_interleave(case.neighbors)
    edge_id = torch.arange(edges, device=device, dtype=torch.int64)
    offset = edge_id.remainder(case.neighbors).add(1)
    source = (destination + offset).remainder(atoms)
    edge_index = torch.stack((source, destination))

    # A deterministic, nonzero geometry avoids special-casing the radial branch.
    phase = edge_id.to(torch.float32) + float(seed)
    edge_vec = torch.stack(
        (
            torch.sin(phase * 0.013),
            torch.cos(phase * 0.017),
            torch.sin(phase * 0.019 + 0.5),
        ),
        dim=-1,
    ).to(device=device)
    edge_vec = edge_vec + 0.05 * torch.randn(
        (edges, 3),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    edge_mask = torch.ones(edges, device=device, dtype=torch.bool)
    destination_order = torch.arange(edges, device=device, dtype=torch.int64)
    destination_row_ptr = torch.arange(
        0,
        edges + 1,
        case.neighbors,
        device=device,
        dtype=torch.int64,
    )
    # A source CSR is useful for force/virial reductions and costs no extra
    # model state in the descriptor path.
    source_order = torch.argsort(source, stable=True)
    source_counts = torch.bincount(source, minlength=atoms)
    source_row_ptr = torch.cat(
        (
            torch.zeros(1, device=device, dtype=torch.int64),
            torch.cumsum(source_counts, dim=0),
        )
    )
    graph_type = torch.arange(
        atoms,
        device=device,
        dtype=torch.int64,
    ).remainder(case.active_types)
    from deepmd.dpmodel.utils.neighbor_graph import (
        NeighborGraph,
    )

    graph = NeighborGraph(
        n_node=torch.tensor([atoms], device=device, dtype=torch.int64),
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=edge_mask,
        destination_order=destination_order,
        destination_row_ptr=destination_row_ptr,
        source_order=source_order,
        source_row_ptr=source_row_ptr,
        destination_sorted=True,
    )
    return SyntheticGraph(graph=graph, atype=graph_type)


def _build_descriptor(case: BenchmarkCase, device: str, seed: int) -> Any:
    """Construct a random DPA4C descriptor matching the case profile."""
    torch = _require_torch()
    from deepmd.pt_expt.descriptor.dpa4c import (
        DescrptDPA4C,
    )

    descriptor = DescrptDPA4C(
        rcut=6.0,
        ntypes=case.model_types,
        channels=case.model.channels,
        lmax=case.model.lmax,
        n_radial=16,
        radial_modes=case.model.radial_modes,
        precision="float32",
        seed=seed,
    ).to(device)
    descriptor.eval()
    return descriptor


def _compressed_forward(
    descriptor: Any,
    workload: SyntheticGraph,
    case: BenchmarkCase,
    *,
    canonical: bool,
) -> Any:
    """Run the compressed reference or compiled generic graph operator."""
    from deepmd.kernels.cuda.dpa4c.graph_compress import (
        _cpu_forward,
        compressed_operator_arguments,
        dpa4c_graph_compress,
        op_available,
    )

    artifacts = _prepare_compression(descriptor, case)

    if canonical:
        from deepmd.kernels.cuda.dpa4c.canonical import (
            _cpu_forward as canonical_forward,
        )
        from deepmd.kernels.cuda.dpa4c.canonical import (
            op_available as canonical_op_available,
        )

        if workload.graph.edge_vec.is_cuda and canonical_op_available():
            if not descriptor.compress:
                descriptor._set_compression(artifacts)
            torch = _require_torch()
            return torch.ops.deepmd.dpa4c_canonical_compress(
                workload.graph.edge_vec,
                workload.graph.edge_index[0],
                workload.graph.destination_row_ptr,
                workload.atype,
                *compressed_operator_arguments(descriptor),
                int(descriptor.lmax),
                *descriptor._compression_scalars,
            )[0]
        info = artifacts["info"].detach().cpu().tolist()
        empty_spin = artifacts["spin_type"].new_empty(0)
        return canonical_forward(
            workload.graph.edge_vec,
            workload.graph.edge_index[0],
            workload.graph.destination_row_ptr,
            workload.atype,
            *(
                artifacts[name]
                for name in (
                    "data",
                    "pair_film",
                    "pair_mixing",
                    "type_embedding",
                    "readout_matrices",
                    "coupling_meta",
                    "coupling_entry",
                    "coupling_value",
                    "output_mean",
                    "output_inv_std",
                )
            ),
            empty_spin,
            artifacts["spin_pair"],
            artifacts["spin_type"],
            int(descriptor.lmax),
            *info,
        )[0]

    if workload.graph.edge_vec.is_cuda and op_available():
        if not descriptor.compress:
            descriptor._set_compression(artifacts)
        return dpa4c_graph_compress(
            descriptor,
            workload.graph,
            workload.atype,
        )

    info = artifacts["info"].detach().cpu().tolist()
    empty_spin = artifacts["spin_type"].new_empty(0)
    return _cpu_forward(
        workload.graph.edge_vec,
        workload.graph.edge_index,
        workload.graph.edge_mask,
        workload.graph.destination_order,
        workload.graph.destination_row_ptr,
        workload.atype,
        *(
            artifacts[name]
            for name in (
                "data",
                "pair_film",
                "pair_mixing",
                "type_embedding",
                "readout_matrices",
                "coupling_meta",
                "coupling_entry",
                "coupling_value",
                "output_mean",
                "output_inv_std",
            )
        ),
        empty_spin,
        artifacts["spin_pair"],
        artifacts["spin_type"],
        False,
        int(descriptor.lmax),
        *info,
    )[0]


def _prepare_compression(descriptor: Any, case: BenchmarkCase) -> dict[str, Any]:
    """Build compression artifacts before a no-grad timing range begins."""
    artifacts = getattr(descriptor, "_benchmark_artifacts", None)
    if artifacts is not None:
        return artifacts
    torch = _require_torch()
    from deepmd.kernels.cuda.dpa4c.graph_compress import (
        build_compression_artifacts,
    )

    with torch.enable_grad():
        artifacts = build_compression_artifacts(
            descriptor,
            case.table_spacing,
        )
    descriptor._benchmark_artifacts = artifacts
    return artifacts


def _forward(
    descriptor: Any,
    workload: SyntheticGraph,
    case: BenchmarkCase,
) -> Any:
    """Dispatch one model path while preserving the fused logical range."""
    with nvtx_range(f"dpa4c:{case.path}"):
        if case.path == "uncompressed":
            return descriptor.call_graph(workload.graph, workload.atype)[0]
        return _compressed_forward(
            descriptor,
            workload,
            case,
            canonical=case.path == "compressed_canonical",
        )


def _execute(
    descriptor: Any,
    workload: SyntheticGraph,
    case: BenchmarkCase,
) -> Any:
    """Execute forward and optional first/second derivative work."""
    torch = _require_torch()
    edge_vec = workload.graph.edge_vec
    if case.mode == "energy":
        with torch.no_grad():
            return _forward(descriptor, workload, case)
    edge_vec.requires_grad_(True)
    output = _forward(descriptor, workload, case)
    gradient = torch.autograd.grad(
        output.square().sum(),
        edge_vec,
        create_graph=case.mode == "energy_force_virial",
        retain_graph=case.mode == "energy_force_virial",
    )[0]
    if case.mode == "energy_force_virial":
        torch.autograd.grad(
            gradient.square().sum(),
            edge_vec,
            allow_unused=True,
        )
    return output, gradient


def run_synthetic_case(
    case: BenchmarkCase,
    *,
    device: str = "cpu",
    seed: int = 17,
    warmup: int = 1,
    repetitions: int = 3,
    environment: EnvironmentMetadata | None = None,
) -> dict[str, Any]:
    """Run one descriptor case and return a normalized result row."""
    case.validate()
    if case.path == "lammps":
        estimate = estimate_workload(case)
        return result_row(
            case,
            estimate,
            environment=environment,
            status="external",
            error="use the LAMMPS adapter for path=lammps",
        )
    if case.precision != "float32":
        estimate = estimate_workload(case)
        return result_row(
            case,
            estimate,
            environment=environment,
            status="not_run",
            error="compressed CUDA benchmark currently requires float32",
        )
    torch = _require_torch()
    graph_timing, workload = measure(
        "graph",
        lambda: build_synthetic_graph(case, device=device, seed=seed),
        warmup=0,
        repetitions=1,
        device=device,
    )
    descriptor = _build_descriptor(case, device, seed)
    if case.path in {"compressed_generic", "compressed_canonical"}:
        _prepare_compression(descriptor, case)
    estimate = estimate_workload(case)
    timing, output = measure(
        "step",
        lambda: _execute(descriptor, workload, case),
        warmup=warmup,
        repetitions=repetitions,
        device=device,
    )
    del output
    timing_fields = {
        **graph_timing.as_dict(),
        **timing.as_dict(),
        "time_ms": timing.time_ms,
        "repetitions": timing.repetitions,
        "launches": timing.launches,
        "edges_per_second": case.edges / (timing.time_ms * 1.0e-3),
        "atoms_per_second": case.atoms / (timing.time_ms * 1.0e-3),
        "gedge_per_second": case.edges / (timing.time_ms * 1.0e6),
        "logical_phases": "graph,model_fused,backward"
        if case.mode != "energy"
        else "graph,model_fused",
        "phase_split": "fused",
    }
    return result_row(
        case,
        estimate,
        environment=environment,
        status="ok",
        timing=timing_fields,
    )
