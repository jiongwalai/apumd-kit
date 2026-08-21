# DPA4C → NvNMD pre-silicon benchmark

This directory is a measurement scaffold, not an RTL or CUDA implementation.
It keeps the benchmark contract, analytical workload accounting, optional
PyTorch graph runner, LAMMPS command adapter, precision emulator, and plotting
utilities together while leaving the existing DPA4C kernels unchanged.

## Contract

`configs/matrix.json` is the source of the first benchmark matrix. The
default command targets **DPA4C-Air** and uses the exact descriptor profile
derived by `descriptor_profile()`:

| model | `(C0, L, R)`  | moment width | output width |
| ----- | ------------- | -----------: | -----------: |
| Nano  | `(8, 2, 0)`   |           40 |           70 |
| Mini  | `(32, 2, 0)`  |           76 |          144 |
| Neo   | `(32, 2, 4)`  |           76 |          144 |
| Air   | `(64, 3, 4)`  |          115 |          219 |
| Plus  | `(128, 3, 4)` |          223 |          541 |

Cases are intentionally generated as one-dimensional sweeps. `--sweep all`
concatenates atom, neighbor, active-type, mode, tile, and radial-spacing
sweeps while holding the other dimensions at the explicit baseline values in
the matrix; it does not construct the full Cartesian product.

The compressed operator currently requires `type_count > 1`. A case with
`active_types=1` therefore retains that active-species value but uses
`model_types=2` and must not be reported as a true one-row model table.

## Analytical estimates

```bash
python -m benchmarks.dpa4c_nvmd estimate \
    --model Air --sweep all \
    --output /tmp/dpa4c-air-estimates.csv \
    --json-output /tmp/dpa4c-air-estimates.json
```

The estimate records theoretical compulsory traffic and arithmetic:
`FLOP/edge`, `FLOP/atom`, `bytes/edge`, `bytes/atom-step`, model/cache
storage, CSR traffic, workspace, saved-edge traffic, and recompute-vs-save
traffic. These are not substitutes for Nsight counters.

## Synthetic graph runner

The runner builds a deterministic regular directed graph and can execute the
uncompressed PyTorch reference and the compressed generic/canonical reference
paths. If a compiled CUDA operator is available, the generic path dispatches
to it; otherwise the compressed reference remains useful for correctness and
CPU phase timing.

```bash
python -m benchmarks.dpa4c_nvmd run \
    --model Air --path compressed_generic --sweep neighbors \
    --device cuda --warmup 5 --repetitions 20 \
    --output /tmp/dpa4c-air-generic.csv \
    --json-output /tmp/dpa4c-air-generic.json
```

The current CUDA compressed implementation consumes FP32 tables and weights.
Non-FP32 contract values are accepted for analytical/precision campaigns and
are reported as `not_run` by the execution runner until a software emulator or
hardware implementation supplies the candidate arithmetic.

Logical timing ranges intentionally label fused work as `model_fused`; they do
not pretend that radial lookup, harmonics, aggregation, invariant readout, and
MLP are independent kernels. Use Nsight Systems/Compute on representative
points and keep the raw reports outside the repository.

## LAMMPS/Kokkos adapter

Provide an input that loads the appropriate `deepmd/kk` model and keeps the
neighbor/force/virial path device-resident:

```bash
python -m benchmarks.dpa4c_nvmd run \
    --model Air --path lammps \
    --lammps-input /abs/path/in.lammps \
    --lammps-command lmp \
    --output /tmp/dpa4c-air-lammps.csv
```

The adapter invokes the command without a shell and parses LAMMPS' `Loop time`
line when present. It does not manufacture an Air model; supply an Air frozen
model or a generated case-specific artifact.

## Precision and MD validation hooks

`precision/emulator.py` provides:

- the first `(Δr, bits) = (0.001, 0.002, 0.005, 0.01) × (32, 24, 16, 12)`
  LUT sweep;
- symmetric fixed-point, FP24, FP16, bfloat16, and integer emulation;
- E/F/V MAE, force RMSE, force maximum, and virial error;
- rotation/reflection checks, finite-difference force checks, and NVE/NVT/NPT
  trajectory summaries.

The hooks accept arrays or callbacks so they can be connected to a frozen
`pt_expt` model without importing PyTorch in the contract/estimate path.

## Eight output figures

```bash
python -m benchmarks.dpa4c_nvmd plot \
    --input /tmp/dpa4c-air-generic.json \
    --output-dir /tmp/dpa4c-air-figures
```

This creates:

1. whole-step breakdown;
1. roofline points;
1. throughput vs atoms;
1. throughput/Gedge/s vs neighbors;
1. runtime vs `(C0, L, R)`;
1. memory traffic vs atoms/model;
1. quantization error vs bit width;
1. projected FPGA throughput vs bandwidth and PE count.

Plotting imports `matplotlib` lazily because it is not a core project
dependency. Normalized CSV/JSON summaries may be archived; raw Nsight output
and large trajectories should remain external.
