# Sparse-CG and CUDA Graph Research Snapshot

This directory documents the final research snapshot preserved on
`archive/sparse-cg-cuda-graph` and tagged as
`research-sparse-cg-cuda-graph-final` before the production package moves to
the dense Torch reference architecture.

## Preserved scope

- Sparse Torch OSQP operators and conjugate-gradient solves.
- Jacobi preconditioning and reduced-system matrix-vector products.
- CUDA Graph capture and replay experiments.
- Adaptive rho, Ruiz scaling, polishing, and warm-state experiments.
- Runtime and end-to-end PyGRANSO benchmark drivers.
- OSQP adapter tests, presentation material, and supporting documentation.

## Baseline validation

Validated on 2026-06-23 (America/Chicago) with Python 3.12 and PyTorch
2.11.0+cu128:

```text
python -m pytest test_osqp_torch_adapter.py -q -p no:cacheprovider
79 passed, 10 warnings in 40.37s
```

The warnings were limited to PyTorch sparse beta/invariant notices and OSQP
deprecation notices. No test failed.

## Archive policy

This snapshot is research history, not the supported production solver path.
Future experiments should branch from the archive reference rather than add
custom numerical linear-solver code back to the normal package path.
