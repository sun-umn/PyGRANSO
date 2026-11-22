# Pinned OSQP upstream test coverage (double precision)

The 18 README replays exercise nine QPs twice, with polishing off and on. They check exact public result bytes. They do not exercise the full OSQP test suite. For example, replaying the initial matrix-update fixture never calls a matrix update. The separate update tests now call the native API and solve the changed problem on both Torch devices.

- [All 71 cases](osqp-upstream-cases.csv) lists the fixed upstream source line, overlap with the 18 replays, native results by backend, executable Torch counterparts and remaining gaps.
- [All 144 named sections](osqp-upstream-sections.csv) records where each section actually ran and the available Torch comparison. A related forward solve does not exercise an adjoint or a code generator.
- [Execution receipts](osqp-upstream-double-results.json) identify the native binaries, failures and exact pytest node IDs. Counts below are per selection.

## What ran

The inventory counts uncommented Catch cases and named sections under the pinned [OSQP tests tree](https://github.com/osqp/osqp/tree/1572ae068e9ce9ca723cf8223548ade1ff7acc29/tests). All 71 cases and all 144 sections appeared in the native reports across configurations. The test build registered the tracked demo and no-active-set files explicitly. The commented-out test_solve_linsys contributes no executable case. Build/install checks and the separate QDLDL project are outside the 71-case denominator.

| Original native double executable | Cases passed / executed | Assertions passed / executed |
|---|---:|---:|
| Builtin solver | 40 / 40 | 551 / 551 |
| Builtin solver with custom allocator | 40 / 40 | 551 / 551 |
| Builtin linear algebra | 28 / 28 | 177 / 177 |
| CUDA solver | **19 / 20** | **313 / 314** |
| CUDA linear algebra | 24 / 24 | 172 / 172 |

The native CUDA failure is **Nonconvex: Solve**. At the [status assertion](https://github.com/osqp/osqp/blob/1572ae068e9ce9ca723cf8223548ade1ff7acc29/tests/non_cvx/test_non_cvx.cpp#L73), native CUDA returned **7** (iteration limit). Builtin C and Torch on CPU, CUDA tensors with QDLDL, and CUDA with cuDSS returned **9** (nonconvex). Both statuses describe unsuccessful solves. This is a mismatch against the native CUDA backend. The experiment does not establish its first internal cause. Torch's default QDLDL route uses CPU factorization and triangular solves even with CUDA tensors.

Expected failures were also tested. At sigma = 1e-6 the builtin setup returned error 4 and both Torch devices raised the inertia error. At sigma = 5 setup succeeded and the subsequent solve reported 9. The three invalid bound updates returned native error 1 and preserved the prior data. The Torch calls raised before replacing committed information. Tests for primal and dual infeasibility compare the actual double status to the builtin C status even when the upstream assertion accepts either one.

## Torch coverage and remaining gaps

| Evidence available for a distinct upstream case | Cases |
|---|---:|
| Paired solver observable, with the API/field limits in the case table | 20 |
| Builtin agreement with a native CUDA status divergence | 1 |
| Partial or semantic comparison of solver behavior or algebra | 24 |
| Numerical tensor controls on upstream algebra data | 7 |
| Only a related setup or forward solve on Torch | 8 |
| Native only | 11 |
| **Total** | **71** |

The 11 native-only cases are the codegen defines defaults, four CSC constructors (zero, scalar diagonal, vector diagonal, identity), uninitialized adjoint workspace, four vector creation cases (float/int allocation and float/int device-pointer wrappers), and settings-struct defaults. The eight related-only cases are the other seven code-generation cases and the adjoint basic test. The individual CSV rows give their original names. These outputs and error paths were tested in C; PyGRANSO exposes no equivalent operation. For matrix equality, subvector operations and the diagonal of A-transpose D A, the numeric controls exercise managed tensors. They do not validate the C memory contract, indirect solver or derivatives.

Other partial boundaries include standalone vector/settings update returns, CSC slot maps, delayed solution allocation and wall-clock timing. A passing case row only covers the fields checked by its linked test functions. The nine CSC matrix update sections additionally compare the three native index/count encodings to whole-matrix resubmission on a retained Torch workspace. Comparisons of updated solutions use the upstream double tolerance of 1e-4. The 18 byte replays use exact equality.

The mapped double selection passed **356**, with **162 deselected**. The additional settings, update, algebra and certificate selection passed **115**. They share 25 nodes (**446 distinct passed**). Both selections had zero failures, skips and collection errors on the audit machine. Host-side setting validation is counted once. The first selection includes all 18 cases on each of CPU, CUDA tensors and the live builtin DLL. Eight standalone upstream codegen/embedded/install executables also exited successfully; the shared-link executable required the installed DLL directory on PATH. These are native build checks.

## Reproduction

The audit used Windows, Python 3.12.10, PyTorch 2.14.0+cu130.cudss071, CUDA 13.0, a GeForce GTX 1650, SciPy 1.17.1 and optional Python OSQP 1.1.3. The native reference for live paired tests is the pinned double/64-bit-index DLL with SHA-256 97d14c66f40b471e3220454374d3bcfc429c64e6297b0062b6f60a0264e043c9. Its location and hash gate are in [the oracle](../tests/pinned_osqp_c_oracle.py). The optional Python package supplies separate reference tests.

Follow the [README](../README.md#reproduce-the-18-double-precision-results) for the portable 18-case replay. The following additional modules use double precision. Algebra and certificate controls can run without the native DLL. The update module skips live comparisons if that DLL is absent.

```sh
python -m pytest -q tests/test_upstream_double_updates.py tests/test_upstream_double_algebra.py tests/test_torch_osqp_infeasibility.py
```

On an equivalent machine with the optional dependencies, both devices and the verified DLL, replay the exact mapped selection from the repository root. Node IDs that include cuDSS require a cuDSS-enabled PyTorch build.

```sh
python -c "import json, pytest, torch; torch.set_num_threads(1); torch.set_default_dtype(torch.float64); nodes=json.load(open('docs/osqp-upstream-double-results.json'))['mapped_nodes']; raise SystemExit(pytest.main(['-q', *nodes]))"
```

For the native inventory, each identified executable was run with Catch arguments `-r xml -o results.xml`, using separate outputs. A case absent from a backend's binary is marked *not registered* in the table. Hosted CPU CI alone cannot reproduce live Windows DLL or GPU results.
