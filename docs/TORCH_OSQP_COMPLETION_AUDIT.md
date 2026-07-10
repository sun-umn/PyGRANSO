# Torch-OSQP Completion Audit

Date: 2026-07-04
Scope: revised dense Torch reference pipeline and release gates

This audit separates implemented behavior from local evidence and external
release gates. A configured gate is not reported as passing until its runner
has produced evidence.

## Architecture and migration

| Requirement | Authoritative evidence | Status |
| --- | --- | --- |
| Preserve research snapshot | Branch `archive/sparse-cg-cuda-graph` exists locally and on `origin` at commit `da142c1`; baseline 79-test result is recorded in the edit log | Implemented and pushed to origin |
| Signed archive tag | Tag `research-sparse-cg-cuda-graph-final` verifies as a PGP-signed tag at `da142c1` with key `913CC0E29352B362D9116C59387554B041B0ACDD`; `origin` has tag object `d40a6e0` | Implemented and pushed to origin |
| Remove research execution from active path | Legacy benchmark, presentation, old adapter test, custom CG, sparse operator, and CUDA Graph code absent from the feature branch; archive branch retains them | Implemented |
| Dense private LU lifecycle | `torchLinearSolve.py` uses `lu_factor_ex`, `lu_solve`, finite/status checks, RHS normalization, reuse, and optional diagnostics | Implemented and unit tested |
| Per-run state | `TorchOSQPWorkspace` is created by each `AlgBFGSSQP` and owns Torch/builtin state, signatures, scaling, factors, and diagnostics | Implemented and unit tested |
| Invalidation contract | Structure, order signature, dimensions, dtype, device, and backend reset state; compatible value updates retain warm state and refactor | Implemented and unit tested |
| Roadmap and Phase 2.1 data contract | `FULL_DEVELOPMENT_AND_VALIDATION_PIPELINE.md` section 7 now maps each milestone to inputs, outputs, data structures, functions, validation, and exit criteria; Phase 2.1 details `TorchOSQPWorkspace`, `DenseLUSolver`, and `LinearSolveDiagnostics` | Implemented |

## Numerical and public behavior

| Requirement | Authoritative evidence | Status |
| --- | --- | --- |
| Preserve KKT/ADMM/projection/dual/residual equations | `torchOSQP.py` plus `test_torch_osqp_kkt.py` | Implemented |
| Ruiz scaling, adaptive rho, warm starts, strict polishing | Feature tests and direct-solver tests | Implemented |
| Public algebra set is exactly auto/builtin/torch | Adapter validation and policy tests | Implemented |
| CPU auto policy | Builtin selection test | Implemented |
| Explicit Torch never changes backend | Explicit unsolved/error tests | Implemented |
| Auto selection and runtime fallback telemetry | Exception, unsolved-status, unsupported-device, memory, and CUDA policy tests | Implemented |
| MPS float64 behavior | Synthetic hardware-independent test verifies builtin CPU float64 result | Implemented; MPS remains unclaimed |
| Dense limit and memory preflight | `n + m <= 2400`, warning/attempt behavior, selection tests, benchmark envelope check | Implemented |
| Common defaults | Adapter defaults and rendered specification | Implemented |
| Validation and symmetry contract | Invalid shape/dtype/device/NaN/bounds/asymmetry tests and optional eigenvalue diagnostic | Implemented |
| No false infeasibility claims | Torch status vocabulary is solved/max-iteration only; hard QP errors propagate to PyGRANSO fallback contracts | Implemented |

Float64 is the authoritative path through estimated KKT conditioning around
`1e8`. Float32 uses the requested `1e-5` tolerances but is qualified only near
estimated KKT conditioning `1e2`. A diagnostic mixed-precision solve showed
that casting solutions and duals back to float32 still violated stationarity
tolerances in 7/10 cases at condition `1e4`, 9/10 at `1e6`, and 10/10 at
`1e8`. Claiming the float64 envelope for returned float32 values would therefore
be unsupported by the requested numerical contract.

## Validation and evidence

| Gate | Evidence | Status |
| --- | --- | --- |
| Deterministic/unit/differential/metamorphic/PyGRANSO | Local pytest suite | Passing locally |
| Windows CPU float64 | Local manifest-backed evidence: 300 rows; 200 condition-qualified release-gate passes and 100 non-gating stress passes | Passing locally |
| Windows CPU qualified float32 | Local manifest-backed evidence: 300 rows; 199 condition-qualified release-gate passes and 101 non-gating rows, including 100 expected stress failures | Passing locally |
| NVIDIA CUDA float64 | Earlier 300-row correctness evidence passed at `02e24fb`; current-source smoke evidence passed 2 rows, but 100-seed local GTX 1650 reruns exceeded the two-hour wrapper | Unpromoted; full current CUDA gate requires representative hardware |
| NVIDIA CUDA qualified float32 | Earlier supported-family correctness evidence passed 200 rows at `02e24fb`; full stress remained non-gating/unpromoted | Unpromoted; representative promotion gate still required |
| NVIDIA CUDA performance | B1/B2/B3 end-to-end medians 12.48x, 21.33x, and 43.46x builtin CPU | Failed; backend unpromoted |
| Linux/Windows/macOS CPU matrix | GitHub Actions run `28557162633` on branch `feature/torch-osqp-dense-reference`; fork `main` nightly run `28767699164` completed all Linux/Windows/macOS float32/float64 buckets | Passing on fork |
| PyTorch 2.8 and current stable | Core workflow run `28557162633`, including Python 3.10-3.13 endpoints | Passing on origin fork |
| Nightly 100-seed platform buckets | Fork `main` workflow run `28767699164`: 1800 total CPU cases, zero release-gate failures; float64 passed 900/900, float32 supported rows passed with 300 stress-only non-gating failures and reproduction files | Passing on fork; upstream default-branch registration still pending |
| CUDA real-hardware promotion | Manual self-hosted correctness, stress, and 5x workflow exists on the feature branch; local GTX 1650 evidence is insufficient for promotion and 5x performance gate fails | Configured; CUDA remains unpromoted |
| ROCm and Apple MPS | No real runner | Unclaimed by design |

Every stability bucket writes a case CSV, environment/settings/seed manifest,
Markdown summary, and one serialized QP per failure. Provenance is captured
before output creation so the dirty flag describes source state. Each manifest
records the exact commit, source hash, platform, settings, and seeds for its
own run; the audit intentionally treats those manifests as the authoritative
provenance rather than hard-coding a commit that would become stale after a
documentation-only or renderer-only follow-up. Earlier CUDA correctness
evidence remains useful regression evidence, but CUDA is not promoted until
representative current-source hardware gates pass.
Time-limit exits now write the same artifact set for completed cases, mark the
manifest as `timed_out` and `partial_results`, record the last completed case,
and exit nonzero so a bucket can fail without losing local telemetry.
Conditioning is estimated by a deterministic dense CPU KKT diagnostic with the
common initial `rho` and `sigma`; backend solves still run on the requested
device, but the evidence gate does not spend accelerator time on diagnostic SVD
kernels.

## Documentation and reporting

| Deliverable | Status |
| --- | --- |
| Two-part decision-complete Markdown specification | Implemented |
| Roadmap-style milestone plan with clear data structures, inputs, outputs, functions, validation, and exit criteria | Implemented; `docs/plans/roadmap.md` is the checked/unchecked status source and links to the 14 phase plans |
| Rendered PDF with TOC, support/risk tables, decision log, and controlled breaks | Implemented and visually inspected |
| Dense benchmark and B1/B2/B3 performance gate | Implemented |
| Code-edit log | Maintained at `.codex/code-edit-log.md` |
| Release-readiness tracker | Maintained at `F:\UMN Researches\Ju Research\Report\2026-07-04_pygranso_torch_osqp_release_tracking.md` |

## Remaining release actions (PR Deferred)

1. Obtain upstream review/merge permission for PR `#63` (Deferred: Do not do PR until project is perfect).
2. Merge/register the feature-branch-only nightly and CUDA workflows on upstream
   `main` before relying on upstream schedules/dispatch (Deferred).
3. Keep CUDA unpromoted until its representative end-to-end median is no worse
   than 5x builtin CPU OSQP.
4. Obtain ROCm and MPS runners before making either support claim.
