# SPDX-License-Identifier: LGPL-3.0-or-later
"""Analytical DPA4C workload accounting for NvNMD design-space studies."""

from __future__ import (
    annotations,
)

from dataclasses import (
    asdict,
    dataclass,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

MODEL_TYPE_CAPACITY = 118

if TYPE_CHECKING:
    from .contract import (
        BenchmarkCase,
        ModelSpec,
    )


@dataclass(frozen=True)
class ProfileDimensions:
    """Dimensions derived from the compiled DPA4C descriptor profile."""

    degree_channels: tuple[int, ...]
    ranks: tuple[int, ...]
    moment_width: int
    gram_width: int
    bispectrum_width: int
    quartic_width: int
    output_width: int

    def as_dict(self) -> dict[str, Any]:
        """Return the profile widths as a JSON-compatible mapping."""
        return asdict(self)

    @property
    def state_width(self) -> int:
        """Return moments plus the two saved normalizers."""
        return self.moment_width + 2


@dataclass(frozen=True)
class WorkloadEstimate:
    """Analytical FLOP and compulsory-traffic estimate for one case."""

    atoms: int
    edges: int
    flops_forward: int
    flops_backward: int
    flops_total: int
    flops_per_edge: float
    flops_per_atom: float
    edge_bytes: int
    node_bytes: int
    csr_bytes: int
    model_bytes: int
    model_cache_bytes: int
    edge_intermediate_bytes: int
    saved_edge_bytes: int
    workspace_bytes: int
    ghost_bytes: int
    total_step_bytes: int
    recompute_step_bytes: int
    save_step_bytes: int
    bytes_per_edge: float
    bytes_per_atom_step: float
    arithmetic_intensity: float
    profile: ProfileDimensions

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible estimate."""
        result = asdict(self)
        result["profile"] = asdict(self.profile)
        return result


def _degree_channels(channels: int, lmax: int) -> tuple[int, ...]:
    """Mirror the backend-neutral fixed degree profile without importing torch."""
    exponent = channels.bit_length() - 1
    degree_one = max(4, 1 << ((exponent + 1) // 2))
    degree_two = max(4, degree_one >> 1)
    return (channels, degree_one, degree_two, *([1] * (lmax - 2)))


def _allowed_triples(lmax: int) -> tuple[tuple[int, int, int], ...]:
    """Mirror the sorted O(3)-even degree triples in the backend layout."""
    return tuple(
        (first, second, third)
        for first in range(1, lmax + 1)
        for second in range(first, lmax + 1)
        for third in range(second, lmax + 1)
        if third <= first + second and (first + second + third) % 2 == 0
    )


def _triple_outputs(
    triple: tuple[int, int, int],
    ranks: tuple[int, ...],
) -> int:
    """Return independent probe contractions for one degree triple."""
    first, second, third = triple
    first_rank, second_rank, third_rank = (
        ranks[first - 1],
        ranks[second - 1],
        ranks[third - 1],
    )
    if first == third:
        return first_rank * (first_rank + 1) * (first_rank + 2) // 6
    if first == second:
        return first_rank * (first_rank + 1) // 2 * third_rank
    if second == third:
        return first_rank * second_rank * (second_rank + 1) // 2
    return first_rank * second_rank * third_rank


def profile_dimensions(model: ModelSpec) -> ProfileDimensions:
    """Return the exact profile widths used by the DPA4C implementation.

    The formulas mirror ``deepmd.dpmodel.descriptor.dpa4c_nn`` and the CUDA
    ``Profile`` structure. Keeping this calculation dependency-free lets the
    benchmark contract run before importing PyTorch.
    """
    model.validate()
    degree_channels = _degree_channels(model.channels, model.lmax)
    ranks = (degree_channels[2], 2, *([1] * (model.lmax - 2)))
    moment_width = sum(
        (2 * degree + 1) * width for degree, width in enumerate(degree_channels)
    )
    gram_width = sum(width * (width + 1) // 2 for width in degree_channels[1:])
    bispectrum_width = sum(
        _triple_outputs(triple, ranks) for triple in _allowed_triples(model.lmax)
    )
    quartic_width = ranks[0] * ranks[1]
    bispectrum_base = degree_channels[0] + gram_width
    output_width = (
        bispectrum_base + bispectrum_width + quartic_width + model.channels + 2
    )
    return ProfileDimensions(
        degree_channels=degree_channels,
        ranks=ranks,
        moment_width=moment_width,
        gram_width=gram_width,
        bispectrum_width=bispectrum_width,
        quartic_width=quartic_width,
        output_width=output_width,
    )


def _edge_flops(model: ModelSpec, profile: ProfileDimensions) -> int:
    """Estimate forward arithmetic for one directed edge."""
    radial_values = model.channels + model.radial_modes
    table = 8 * radial_values
    film = 2 * model.channels
    mixing = 2 * model.channels * model.radial_modes
    harmonics = 4 * (model.lmax + 1) ** 2
    moment_components = sum(
        (2 * degree + 1) * width
        for degree, width in enumerate(profile.degree_channels)
        if degree > 0
    )
    aggregation = 4 * moment_components + 3
    return table + film + mixing + harmonics + aggregation


def _node_flops(model: ModelSpec, profile: ProfileDimensions) -> int:
    """Estimate normalization, invariant readout, and fitting arithmetic."""
    gram = sum(
        (2 * degree + 1) * width * (width + 1)
        for degree, width in enumerate(profile.degree_channels)
        if degree > 0
    )
    probe = sum(
        2 * width * rank
        for width, rank in zip(
            profile.degree_channels[1:],
            profile.ranks,
            strict=True,
        )
    )
    invariant = 2 * (gram + probe + profile.output_width)
    hidden = model.fitting_width
    fitting = 2 * (profile.output_width * hidden + 2 * hidden * hidden + hidden)
    activation = 8 * hidden * 3
    return 32 + invariant + fitting + activation


def estimate_workload(
    case: BenchmarkCase,
    *,
    dtype_bytes: int = 4,
    index_bytes: int = 4,
    save_edge_intermediates: bool = False,
) -> WorkloadEstimate:
    """Estimate FLOP, memory traffic, and workspace for a benchmark case.

    The result is a compulsory-traffic model, not a replacement for Nsight
    counters. It intentionally exposes every term so measured DRAM/L2 traffic
    can be used to calibrate the model later.
    """
    case.validate()
    if dtype_bytes <= 0 or index_bytes <= 0:
        raise ValueError("dtype_bytes and index_bytes must be positive")
    profile = profile_dimensions(case.model)
    edges = case.edges
    edge_flops = _edge_flops(case.model, profile)
    node_flops = _node_flops(case.model, profile)
    forward = edges * edge_flops + case.atoms * node_flops
    backward = forward if case.mode != "energy" else 0
    total_flops = forward + backward

    # Generic graph ABI: edge vectors, two endpoints, mask, destination order.
    if case.path == "compressed_canonical":
        # Canonical storage derives destinations and their order from CSR.
        edge_bytes = edges * (3 * dtype_bytes + index_bytes)
    else:
        edge_bytes = edges * (3 * dtype_bytes + 2 * index_bytes + 1 + index_bytes)
    # Node state and descriptor output are written once per model invocation.
    node_bytes = (
        case.atoms * (profile.state_width + profile.output_width) * dtype_bytes
        + case.atoms * index_bytes
    )
    csr_bytes = (case.atoms + 1) * index_bytes
    table_rows = int(6.0 / case.table_spacing) + 1
    table_bytes = (
        table_rows * 6 * (case.model.channels + case.model.radial_modes) * dtype_bytes
    )
    pair_bytes = case.model.channels * (2 + case.model.radial_modes) * dtype_bytes
    active_model_bytes = (
        table_bytes
        + case.model_types * case.model.channels * dtype_bytes
        + case.model_types**2 * pair_bytes
    )
    model_bytes = (
        table_bytes
        + MODEL_TYPE_CAPACITY * case.model.channels * dtype_bytes
        + MODEL_TYPE_CAPACITY**2 * pair_bytes
    )
    edge_feature_width = (
        case.model.channels + case.model.radial_modes + (case.model.lmax + 1) ** 2 + 2
    )
    edge_intermediate_bytes = edges * edge_feature_width * dtype_bytes
    saved_edge_bytes = edge_intermediate_bytes if save_edge_intermediates else 0
    workspace_bytes = profile.state_width * case.atoms * dtype_bytes
    recompute_step_bytes = edge_bytes + node_bytes + csr_bytes
    save_step_bytes = recompute_step_bytes + edge_intermediate_bytes
    total_step_bytes = (
        save_step_bytes if save_edge_intermediates else recompute_step_bytes
    )
    if total_step_bytes <= 0:
        raise ValueError("estimated step traffic must be positive")
    bytes_per_edge = total_step_bytes / edges
    bytes_per_atom_step = total_step_bytes / case.atoms
    arithmetic_intensity = total_flops / total_step_bytes
    return WorkloadEstimate(
        atoms=case.atoms,
        edges=edges,
        flops_forward=forward,
        flops_backward=backward,
        flops_total=total_flops,
        flops_per_edge=total_flops / edges,
        flops_per_atom=total_flops / case.atoms,
        edge_bytes=edge_bytes,
        node_bytes=node_bytes,
        csr_bytes=csr_bytes,
        model_bytes=model_bytes,
        model_cache_bytes=active_model_bytes,
        edge_intermediate_bytes=edge_intermediate_bytes,
        saved_edge_bytes=saved_edge_bytes,
        workspace_bytes=workspace_bytes,
        ghost_bytes=0,
        total_step_bytes=total_step_bytes,
        recompute_step_bytes=recompute_step_bytes,
        save_step_bytes=save_step_bytes,
        bytes_per_edge=bytes_per_edge,
        bytes_per_atom_step=bytes_per_atom_step,
        arithmetic_intensity=arithmetic_intensity,
        profile=profile,
    )
