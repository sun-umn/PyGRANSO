# Full Development and Validation Pipeline

Version: 2.2
Date: 2026-07-04
Status: Release-candidate specification and roadmap; accelerator promotion gated

## Part I - Executive Summary

### 1. Decision

Build a correctness-first, Torch-native OSQP reference route for PyGRANSO.
The implementation is dense and uses PyTorch's built-in LU factorization. It
is not presented as a sparse large-scale solver. Sparse acceleration is a
future backend that must fit behind the same internal interface.

The first trustworthy milestone supports feasible convex QPs and is judged by
observable outcomes: compatible status, feasibility, stationarity, residuals,
and objective. It does not require identical ADMM trajectories or iteration
counts across devices or backends.

### 2. Primary acceptance envelope

| Item | Supported contract |
| --- | --- |
| Problem class | Feasible convex QPs in OSQP form |
| Authoritative precision | float64 |
| Qualified precision | float32 with looser tolerances and KKT conditioning approximately 1e2 |
| Automatic dense limit | KKT dimension n + m <= 2400 |
| Float64 supported conditioning | Approximately 1e8 |
| Stress-only conditioning | Approximately 1e10 |
| Linear algebra | Reusable torch.linalg.lu_factor_ex and lu_solve |
| Required features | Ruiz scaling, adaptive rho, polishing, warm starts |
| Default CPU policy | Builtin OSQP |
| Default accelerator policy | Torch only after that backend passes its gates |
| Failure budget | Zero unexplained failures inside the supported matrix |

### 3. Backend support matrix

| Backend | Precision | Initial status | Promotion evidence |
| --- | --- | --- | --- |
| Torch CPU on Linux | float32/float64 | Release-gated | Core, nightly, end-to-end |
| Torch CPU on Windows | float32/float64 | Local correctness pass; release-gated | Cross-version CI still required |
| Torch CPU on macOS | float32/float64 | Release-gated | Core, nightly, end-to-end |
| NVIDIA CUDA | float32/float64 | Unpromoted | Earlier local correctness evidence passed, but current-source full CUDA reruns on the GTX 1650 exceeded the two-hour local wrapper and 12.48x, 21.33x, and 43.46x end-to-end slowdowns failed the <=5x gate |
| AMD ROCm | float32/float64 | Unclaimed | Real-hardware runner required |
| Apple MPS | float32 only | Unclaimed | Reusable LU must pass on real Apple hardware |
| Apple MPS float64 | Unsupported | Builtin CPU result under auto | MPS does not support float64 tensors |

### 4. Outcome

The default `auto` policy follows the requested optimization device. CPU work
uses builtin OSQP. Accelerator work uses Torch only when the device, KKT size,
memory estimate, correctness suite, and performance sanity gate are all
satisfied. Any automatic Torch exception or unsolved status produces a visible
builtin fallback and a complete causal record. Explicit Torch requests never
change backend silently.

### 5. Main risks and controls

| Risk | Control |
| --- | --- |
| Repeated dense refactorization | Cache LU and refactor only when the KKT matrix changes |
| Dense memory growth | n + m limit plus conservative working-memory preflight |
| Float32 stagnation | Dtype-aware 1e-5 defaults and condition-aware classification |
| Hidden CPU fallback | Warning plus structured backend, trigger, transfer, and result fields |
| Cross-run state contamination | One private workspace per BFGS-SQP run |
| Misclassified infeasibility | Do not claim certificates in the first milestone |
| Optional polishing corrupts a valid result | Accept only nonworse KKT metrics; numerical failure raises |
| Unsupported accelerator claims | Promote each backend only after real-hardware evidence |

<!-- PAGEBREAK -->

## Part II - Decision-Complete Engineering Specification

### 6. Architecture

```text
PyGRANSO BFGS-SQP
  -> steering QP or stationarity QP
  -> solveQP.py
  -> osqpTorchAdapter.py
  -> canonical P, q, A, l, u
     -> builtin CPU OSQP
     -> dense Torch OSQP reference
        -> OSQP ADMM equations
        -> private DenseLUSolver boundary
        -> PyTorch LU factorization and repeated solves
```

The linear-solver boundary is internal. The user selects only
`osqp_algebra={auto,builtin,torch}`. No nested linear-solver selector is
exposed until a second validated Torch backend exists.

### 7. Milestone roadmap

This roadmap is the work-breakdown view of the specification. Each phase is
complete only when its interface, data structures, tests, and evidence artifacts
are all present. The roadmap intentionally keeps solver correctness ahead of
performance and moves certificate/nonconvex/sparse claims to later work.

The decision-complete release checklist and per-phase implementation plans live
under `docs/plans/`, with `docs/plans/roadmap.md` as the status source for
checked and unchecked release-readiness tasks.

#### Phase 0 - Research snapshot and rollback point

Goal: preserve the prior sparse-CG/CUDA Graph research state before narrowing
the main package path to the dense reference route.

Inputs:

- current sparse-CG, sparse-operator, Jacobi, and CUDA Graph research code;
- old benchmarks, tests, notebooks, documentation, and local evidence;
- current git branch and release candidate worktree.

Outputs:

- archive branch `archive/sparse-cg-cuda-graph`;
- signed tag `research-sparse-cg-cuda-graph-final`;
- baseline validation result for the archived snapshot;
- main branch with research execution removed from the active package path.

Data structures and artifacts:

- git branch/tag objects are the durable archive handles;
- archived tests and benchmark files remain available only through the archive;
- `.codex/code-edit-log.md` records the baseline and migration decision.

Functions and commands:

- `git branch`, `git tag`, `git push`;
- package-path deletion of custom CG, sparse operator, Jacobi, and CUDA Graph
  selection code.

Validation:

- verify archive branch and tag resolve to the intended commit;
- verify active package imports no removed research solver path;
- record baseline test status before removal.

Exit criteria:

- a reviewer can recover the research snapshot from git;
- the active route contains no hidden sparse-CG/CUDA Graph execution path.

#### Phase 1 - Public QP contract and backend policy

Goal: define the public behavior before implementing the dense reference
internals.

Inputs:

- PyGRANSO QP arguments `H`, `f`, `A`, `b`, `LB`, `UB`;
- requested `torch_device`;
- `double_precision`;
- `osqp_options` containing `algebra`, `settings`, and `workspace`.

Outputs:

- canonical OSQP problem `P`, `q`, `A_osqp`, `l`, `u`;
- backend selection result: `builtin` or `torch`;
- result tensor on the requested or documented fallback device;
- structured `info` dictionary when `return_info=True`.

Data structures:

- `DEFAULT_OSQP_SETTINGS` is the common settings source;
- `PROMOTED_ACCELERATOR_BACKENDS` controls automatic accelerator eligibility;
- fallback telemetry is stored in `info["fallback"]`;
- backend cache state is stored in `TorchOSQPWorkspace`.

Functions:

- `solveQP(...)` is the PyGRANSO QP entry point;
- `_solve_osqp_with_warm_state(...)` injects per-run workspace state;
- `solve_osqp_torch_qp(...)` owns public OSQP backend selection;
- `_select_backend(...)` applies `auto`, `builtin`, and `torch` policy;
- `_build_constraints_torch(...)` and `_build_constraints_numpy(...)` create
  canonical OSQP rows.

Validation:

- accepted `osqp_algebra` values are exactly `auto`, `builtin`, and `torch`;
- CPU `auto` uses builtin OSQP;
- unpromoted or unsupported accelerator `auto` falls back with warning and
  telemetry;
- explicit `torch` never changes backend silently.

Exit criteria:

- public behavior is stable before inner solver work begins;
- unsupported options fail with actionable migration errors.

#### Phase 2.1 - Data model, workspace, and factorization lifecycle

Goal: make reusable state explicit, private, and safe across repeated QP solves.
This is the primary data-structure milestone.

Primary data structures:

- `TorchOSQPWorkspace`
  - Owner: one `AlgBFGSSQP` run.
  - Input role: passed as `osqp_options["workspace"]`.
  - Output role: records warm state, solver factors, backend cache, and latest
    diagnostics.
  - Warm-state fields: `state["x"]`, `state["z"]`, `state["y"]`.
  - Structure fields: `problem_signature`, `constraint_order_signature`,
    `p_pattern`, `a_pattern`.
  - Scaling fields: `scaling`, `scaling_source_p`, `scaling_source_a`,
    `scaling_passes`.
  - Rho fields: `rho_bar`, `rho_setting`.
  - Backend fields: `active_backend`, `builtin_cache`, `builtin_stats`.
  - Solver field: `linear_solver`, an owned `DenseLUSolver`.
  - Diagnostic field: `last_info`.
- `DenseLUSolver`
  - Owner: `TorchOSQPWorkspace.linear_solver`.
  - Input role: receives dense KKT matrix `K` and finite RHS tensors.
  - Output role: returns solution tensors and `LinearSolveDiagnostics`.
  - Cached fields: matrix clone, LU factors, pivots, factorization status.
  - Counters: `factorization_count`, `solve_count`.
- `LinearSolveDiagnostics`
  - Owner: emitted per linear solve.
  - Fields: solver name, LU info, factorization count, solve count, optional
    absolute and relative linear residuals.

Workspace input/output contract:

| Operation | Input | Preserved output | Invalidated output |
| --- | --- | --- | --- |
| Vector-only update | same P/A values and structure, new q/l/u | x, z, y, rho, scaling, LU | none |
| Matrix-value update | same P/A shape, pattern, dtype, device, order | x, z, y | scaling and LU |
| Rho or sigma change | compatible problem, changed setting | x, z, y | LU |
| Dimension/order/structure change | changed shape, pattern, or constraint order | none | x, z, y, scaling, LU |
| Dtype/device/backend change | changed dtype, device, or selected backend | none | complete workspace |

Function contracts:

| Function | Input | Output | Failure mode |
| --- | --- | --- | --- |
| `TorchOSQPWorkspace.ensure_backend(backend)` | `"builtin"` or `"torch"` | boolean backend-change flag | `ValueError` for unknown backend |
| `TorchOSQPWorkspace.reset_torch()` | none | clears Torch warm state, scaling, rho, LU, diagnostics | none |
| `DenseLUSolver.factorize(K)` | square finite strided float32/float64 tensor | cached LU/pivots and incremented factorization count | `TorchLinearSolveError` or validation error |
| `DenseLUSolver.solve(rhs)` | finite vector or matrix RHS matching K | solution with vector shape restored plus diagnostics | `TorchLinearSolveError` or validation error |
| `DenseLUSolver.factorize_if_needed(K)` | candidate KKT matrix | `True` when factorized, `False` when reused | same as `factorize` |
| `_prepare_workspace(workspace, P, A, order)` | workspace plus canonical matrices | compatible state preserved or reset deterministically | none for valid inputs |

Validation:

- unit-test RHS vector normalization and matrix RHS solves;
- reject nonfinite matrices/RHS values before LU/solve;
- check LU `info` and nonfinite factors/solutions;
- prove factorization reuse for vector-only updates;
- prove refactorization for matrix-value, rho, or sigma updates;
- prove complete invalidation for structure/order/dtype/device/backend changes.

Exit criteria:

- no module global stores Torch OSQP warm state or factors;
- every reusable object has one owner and one invalidation policy;
- diagnostics can explain whether factors were reused, rebuilt, or cleared.

#### Phase 2.2 - Dense Torch ADMM kernel

Goal: implement the direct dense reference solver around the Phase 2.1
factorization boundary while preserving the OSQP equations.

Inputs:

- validated `P`, `q`, `A`, `l`, `u`;
- normalized settings;
- compatible `TorchOSQPWorkspace`.

Outputs:

- solution tensor `x` shaped `(n, 1)`;
- updated workspace state `x`, `z`, `y`;
- `info` with status, residuals, objective, iterations, rho, scaling, and
  factorization counters.

Data structures:

- dense KKT matrix `K`;
- ADMM vectors `x`, `z`, `y`, `x_tilde`, `z_tilde`, `nu`;
- vector-valued `rho_vec`, with equality rows boosted.

Functions:

- `build_kkt_matrix(P, A, sigma, rho_vec)`;
- `build_kkt_rhs(x, z, y, q, sigma, rho_vec)`;
- `recover_z_tilde(z, nu, y, rho_vec)`;
- `admm_vector_update(...)`;
- `solve_torch_osqp_direct(...)`.

Validation:

- KKT block assembly and RHS tests;
- projection and dual-update equation tests;
- finite solution and residual checks;
- solved/max-iteration status handling only.

Exit criteria:

- mathematical equations match the specification;
- numerical failures are raised or reported as unsolved, never as infeasibility.

#### Phase 2.3 - Scaling, adaptive rho, polishing, and warm starts

Goal: add the numerical features required for observable OSQP agreement without
changing the public backend contract.

Inputs:

- Phase 2.2 dense direct solve;
- settings for scaling, adaptive rho, polishing, and warm start;
- existing workspace state when compatible.

Outputs:

- scaled solve whose accepted residuals/objective are reported in original
  coordinates;
- deterministic rho updates and refactorized KKT matrices;
- accepted polished solution or explicit polishing failure;
- reusable warm state for the next compatible solve.

Data structures:

- Ruiz scaling dictionary: `D`, `E`, `cost`, `passes`;
- original-coordinate state dictionary: `x`, `z`, `y`;
- polishing active-set rows and active RHS;
- polishing info fields in `info`.

Functions:

- `_scaling_for_problem(...)`;
- `_scale_problem(...)`;
- `_initial_scaled_state(...)`;
- `_unscale_state(...)`;
- `_adaptive_rho_update(...)`;
- `_polish_solution(...)`;
- `_residuals(...)`.

Validation:

- scaling round-trip tests;
- deterministic adaptive-rho tests;
- polishing active-row and duplicate-row tests;
- warm-start reuse and invalidation tests.

Exit criteria:

- every residual and objective used for acceptance is in original coordinates;
- polishing cannot silently degrade an accepted result;
- compatible warm starts are observable through diagnostics.

#### Phase 3 - Builtin parity, fallback, and public telemetry

Goal: make builtin OSQP and Torch OSQP comparable through one adapter contract.

Inputs:

- PyGRANSO QP form;
- selected backend;
- normalized common settings;
- per-run workspace.

Outputs:

- builtin or Torch solution tensor;
- common residual/objective metrics;
- backend/fallback telemetry;
- builtin workspace setup/update counters.

Data structures:

- builtin OSQP cache with sparse `P`, sparse `A`, OSQP problem, and last result;
- `builtin_stats` with setup, update, rebuild, and cache-hit counts;
- `info["fallback"]` with trigger, selected backend, fallback backend, status
  or exception, and device-transfer flag.

Functions:

- `_solve_builtin_osqp_path(...)`;
- `_builtin_common_metrics(...)`;
- `_dense_polish_builtin(...)`;
- `_selection_fallback(...)`;
- `_accelerator_capability(...)`;
- `_builtin_result_device(...)`.

Validation:

- builtin cache reuse for structurally compatible updates;
- auto Torch exception fallback;
- auto unsolved-status fallback;
- memory and size fallback;
- MPS float64 CPU-result behavior.

Exit criteria:

- automatic fallback is always visible;
- explicit Torch failures remain explicit;
- builtin and Torch comparisons use shared metrics rather than raw iterates.

#### Phase 4 - Validation evidence pipeline

Goal: prove the solver behavior through deterministic, differential,
metamorphic, randomized, end-to-end, and platform gates.

Inputs:

- unit tests under `tests/`;
- fixed randomized seeds;
- backend/device/dtype matrix;
- PyGRANSO B1/B2/B3 workloads.

Outputs:

- passing deterministic core test result;
- nightly 100-seed evidence per family/backend bucket;
- serialized reproduction data for each failure;
- release-gate status for each supported backend.

Data structures and artifacts:

- `torch_osqp_stability_results.csv`;
- `torch_osqp_stability_manifest.json`;
- `torch_osqp_stability_summary.md`;
- failure reproduction `.pt` files;
- GitHub Actions artifacts for core, nightly, and CUDA promotion workflows.

Functions and scripts:

- `torch_osqp_stability.py`;
- `bench_osqp_dense_reference.py`;
- `bench_pygranso_osqp_workloads.py`;
- `.github/workflows/torch-osqp-core.yml`;
- `.github/workflows/torch-osqp-nightly.yml`;
- `.github/workflows/torch-osqp-cuda-promotion.yml`.

Validation:

- deterministic pytest suite on every change;
- nightly fixed-seed CPU stability buckets;
- self-hosted real-hardware accelerator promotion buckets;
- benchmark gate for automatic accelerator promotion.

Exit criteria:

- zero unexplained failures inside the supported size, dtype, conditioning, and
  backend matrix;
- stress rows are classified as stress evidence, not as support claims.

#### Phase 5 - Backend promotion

Goal: turn backend support on only after backend-specific evidence exists.

Inputs:

- Phase 4 evidence package;
- real hardware for each claimed accelerator;
- performance gate output.

Outputs:

- updated backend support matrix;
- promoted/unpromoted accelerator decision;
- public `auto` behavior aligned with the support matrix.

Data structures:

- `PROMOTED_ACCELERATOR_BACKENDS`;
- support matrix in this document;
- release evidence manifests.

Functions:

- `_accelerator_capability(...)`;
- `_select_backend(...)`;
- `bench_pygranso_osqp_workloads.py`.

Validation:

- CPU support gates on Linux, Windows, and macOS;
- CUDA correctness, stress, and no-worse-than-5x end-to-end median runtime;
- ROCm and MPS remain unclaimed until real runners exist.

Exit criteria:

- no backend is promoted by assumption;
- `auto` routes only to backends with current evidence.

#### Phase 6 - Documentation, PDF, and release handoff

Goal: make the implementation reviewable and reproducible.

Inputs:

- final code, tests, workflows, and evidence;
- this Markdown specification;
- completion audit.

Outputs:

- maintained Markdown specification;
- decision-complete `docs/plans` roadmap and phase plans;
- rendered PDF;
- completion audit;
- code-edit report entries;
- release tracking report under `F:\UMN Researches\Ju Research\Report`;
- release PR notes.

Data structures and artifacts:

- `docs/FULL_DEVELOPMENT_AND_VALIDATION_PIPELINE.md`;
- `docs/TORCH_OSQP_COMPLETION_AUDIT.md`;
- `docs/plans/roadmap.md` and linked phase plans;
- `output/pdf/Full Development and Validation Pipeline - Revised.pdf`;
- `.codex/code-edit-log.md`.

Functions and scripts:

- `scripts/render_pipeline_pdf.py`;
- git/CI evidence inspection commands.

Validation:

- Markdown headings and code blocks render cleanly;
- PDF includes table of contents, support matrix, risk table, roadmap,
  decision log, and controlled page breaks;
- audit status matches current source and evidence.

Exit criteria:

- a reviewer can follow the roadmap from public API to data model to validation
  evidence without reading implementation code first.

### 8. Canonical QP contract

Solve:

```text
minimize    0.5 * x' P x + q' x
subject to  l <= A x <= u
```

Requirements:

- P is square, finite, and positive semidefinite within the supported numerical envelope.
- q and A are finite.
- l and u may contain infinity but never NaN, and l <= u elementwise.
- All Torch inputs use the same device and dtype after adapter normalization.
- Only float32 and float64 are supported.
- Near-symmetric P is replaced by 0.5 * (P + P.T) only when its infinity-norm asymmetry is within a dtype-aware tolerance.
- A diagnostic eigenvalue check is available for tests and debugging; it is not paid on every solve.

### 9. Dense linear-solver lifecycle

The private solver owns the matrix and its LU factors.

```text
factorize(K)
  validate square, dtype, device, and finite values
  call torch.linalg.lu_factor_ex(K, check_errors=False)
  reject nonzero info or non-finite factors

solve(rhs)
  normalize a vector RHS to shape (n, 1)
  validate shape, dtype, device, and finite values
  call torch.linalg.lu_solve(LU, pivots, rhs)
  reject non-finite solutions
  optionally report ||Kx-b|| / max(1, ||b||)

refactorize(K)
  run only after P, A, rho, sigma, dtype, or device changes
```

Reuse the same factorization across ADMM iterations and across compatible
updates to q, l, and u. A change to P or A values refactorizes. A change to
dimensions, sparsity pattern, constraint ordering, dtype, device, or backend
invalidates the complete workspace.

### 10. Optimizer-owned workspace

Each BFGS-SQP run owns one private `TorchOSQPWorkspace`. It stores:

- original-coordinate x, z, and y warm state;
- problem and structure signatures;
- cached scaling vectors and cost scale;
- the latest adaptive rho value;
- the dense LU solver and counters;
- the builtin OSQP update workspace;
- the last structured diagnostics.

Independent, nested, or concurrent PyGRANSO runs never share warm state or
factorizations through module globals.

### 11. Preserved OSQP equations

The direct KKT system is:

```text
[ P + sigma I    A'          ] [x_tilde] = [sigma x - q]
[ A             -diag(rho)^-1] [nu     ]   [z - y/rho ]
```

Recover and update:

```text
z_tilde = z + (nu - y) / rho
x_next = alpha * x_tilde + (1-alpha) * x
z_relaxed = alpha * z_tilde + (1-alpha) * z
z_next = project_[l,u](z_relaxed + y/rho)
y_next = y + rho * (z_relaxed - z_next)
```

Stopping residuals are evaluated in original coordinates:

```text
r_primal = ||A x - z||_inf
r_dual   = ||P x + q + A' y||_inf
```

Nonunique problems are accepted by feasibility, stationarity, objective, and
compatible status rather than by matching x exactly.

### 12. Required numerical features

#### 12.1 Ruiz scaling

Use ten deterministic diagonal-equilibration passes, solve the scaled problem,
then unscale x, z, and y. Reuse cached scaling for vector-only parametric
updates. All acceptance residuals and objectives are reported in original
coordinates.

#### 12.2 Adaptive rho

Use a deterministic interval of 50 and an update tolerance of 5. Equality rows
receive the larger vector-valued rho policy. Every accepted rho change rebuilds
and refactorizes K while continuing from the current x, z, and y.

<!-- PAGEBREAK -->

#### 12.3 Polishing

Build the active-set polishing KKT system through the same LU boundary. Reuse
the factorization for refinement steps. A candidate is accepted only when its
KKT metric is no worse or it satisfies the target tolerances. A factorization,
refinement, or candidate-acceptance failure raises when polishing was requested.

#### 12.4 Warm starts

Warm starts are enabled internally. Compatible vector updates reuse x, z, y,
rho, scaling, and LU. Matrix-value updates retain x, z, and y but recompute
scaling and factors. Structural changes clear the workspace.

### 13. Defaults

| Setting | Default |
| --- | ---: |
| rho | 0.1 |
| sigma | 1e-6 |
| alpha | 1.6 |
| max_iter | 4000 |
| check_termination | 25 |
| float64 eps_abs and eps_rel | 1e-8 |
| float32 eps_abs and eps_rel | 1e-5 |
| scaling | 10 |
| adaptive_rho | true |
| rho_update_interval | 50 |
| rho_update_tolerance | 5 |
| polishing | true |
| polish_delta | 1e-6 |
| polish_refine_iter | 3 |
| warm_start | true |

### 14. Backend selection and fallback

| Request | Behavior |
| --- | --- |
| auto on CPU | Builtin CPU OSQP |
| auto on validated accelerator inside limits | Torch reference route |
| auto on unsupported accelerator | Warn and use builtin |
| auto above size or memory envelope | Warn and use builtin |
| auto Torch exception or unsolved status | Warn, retry builtin, retain causal telemetry |
| explicit builtin | Builtin CPU OSQP and return on requested device |
| explicit torch inside limits | Torch reference route |
| explicit torch above limits | Warn and attempt; never silently change backend |

Fallback diagnostics include the requested and selected backends, the original
exception or status, the fallback backend and outcome, and whether data moved
between accelerator and CPU.

### 15. Status and error semantics

- Return structured statuses for solved and maximum-iteration outcomes.
- Raise for invalid inputs, unsupported explicit operations, LU failure, NaN or Inf, and numerical polishing failure.
- Under auto, any unsolved-compatible Torch status is eligible for a warned builtin retry.
- An explicit Torch failure propagates to PyGRANSO's steering or stationarity fallback contract without being rewritten as a secondary unpacking or unbound-variable error.
- Never label a linear solve failure as infeasibility.
- Move primal and dual infeasibility certificates and nonconvex detection to a future milestone.

### 16. Validation pipeline

```text
Dense LU unit tests
  -> KKT assembly and ADMM equation tests
  -> deterministic complete-QP tests
  -> scaling, adaptive-rho, polishing, and warm-state tests
  -> matched-setting builtin-vs-Torch differential tests
  -> PyGRANSO steering and stationarity contracts
  -> metamorphic tests
  -> seeded randomized and conditioning tests
  -> B1/B2/B3 and constrained end-to-end runs
  -> backend-specific hardware gates
  -> performance sanity gate
```

Core CI runs deterministic cases on every change. Nightly validation runs 100
fixed seeds per family and backend bucket with a two-hour budget. Every failure
records the complete QP, seed, settings, environment, results, and residuals.

Inside the supported matrix, the failure budget is zero unexplained failures.
Cases near condition number 1e10 are classified as stress evidence rather than
as guaranteed support.

### 17. Evidence package

Each stability run produces:

- `torch_osqp_stability_results.csv` with case-level gates;
- `torch_osqp_stability_manifest.json` with commit, platform, hardware, Python, PyTorch, OSQP, backend, settings, and seeds;
- `torch_osqp_stability_summary.md` with family totals;
- one serialized reproduction file for every failure.

If the wall-clock budget is exceeded after a case completes, the run still
writes the CSV, manifest, summary, and any failure reproductions collected so
far, marks `timed_out=true` and `partial_results=true` in the manifest, records
the last completed case, and exits nonzero. A timed-out bucket is not passing
release evidence, but it remains reproducible telemetry instead of a silent
artifact loss.

The manifest captures Git provenance before creating its own output directory,
so `git_dirty` describes source state rather than generated evidence files. It
also records status entries and a deterministic SHA-256 over maintained source,
tests, workflows, scripts, and documentation, making dirty-worktree evidence
exactly identifiable before a release commit exists.

Stability reports estimate conditioning with a deterministic dense CPU KKT
diagnostic using the initial common `rho` and `sigma` settings. The QP solves
still execute on the requested backend, but the condition estimate itself does
not call accelerator SVD/condition kernels; this keeps nightly and hardware
evidence buckets inside the two-hour gate.

Differential tests use identical algorithm settings. They compare status,
primal and dual residuals, objective, equality violation, bound violation, and
finite values. Objective gaps use `abs(torch-reference) / max(1, abs(reference))`.

### 18. Performance gate

Performance is not a correctness criterion. It controls only automatic backend
promotion. On representative accelerator-targeted PyGRANSO workloads, the
Torch median end-to-end time, including necessary transfers, must be no worse
than five times builtin CPU OSQP. A backend that fails remains explicit or
unclaimed even when its correctness suite passes.

Current CUDA promotion evidence is intentionally insufficient. Earlier local
NVIDIA correctness buckets passed, and a current-source CUDA float64 smoke
passes, but current-source 100-seed CUDA float64 reruns on the local GTX 1650
exceeded the two-hour wrapper. Independent end-to-end B1, B2, and B3 medians
were 12.48x, 21.33x, and 43.46x the builtin CPU median respectively. CUDA
therefore remains explicit-only and `auto` records a warned
`cuda_not_promoted` builtin fallback.

### 19. Migration sequence

1. Validate and preserve the sparse-CG/CUDA Graph research snapshot.
2. Create and push archive branch `archive/sparse-cg-cuda-graph`.
3. Create and push signed tag `research-sparse-cg-cuda-graph-final`.
4. Remove custom CG, Jacobi, sparse-operator, CUDA Graph, and selection code from the package path.
5. Add DenseLUSolver and optimizer-owned workspace tests.
6. Refactor direct ADMM around reusable LU.
7. Validate scaling, adaptive rho, polishing, and warm starts.
8. Implement the backend policy, migration errors, size guards, and fallback telemetry.
9. Add differential, randomized, hardware, PyGRANSO, and reporting gates.
10. Promote each backend only after its own correctness and performance evidence passes.

### 20. Decision log

| Decision | Rationale |
| --- | --- |
| Dense reference first | Correctness and maintainability are the primary objective |
| Reusable LU instead of solve per iteration | Preserves built-in numerical ownership without repeated refactorization |
| Observable OSQP agreement | Different devices and factorizations need not match trajectories |
| Float64 authoritative | Float32 degrades materially on ill-conditioned KKT systems |
| Feasible convex scope first | Prevents unvalidated certificate claims |
| Auto follows requested device | Avoids unprofitable CPU-to-accelerator transfers |
| Per-run workspace | Prevents cross-run state contamination |
| Backend-by-backend promotion | Support claims require real hardware |
| Five-times performance ceiling | Prevents severe automatic regressions without making speed the success criterion |

### 21. Future work

After the dense reference route passes all applicable gates, a sparse direct or
iterative backend may implement the same factorize/solve/refactorize contract.
Candidates include cuDSS, torch-sla, future PyTorch sparse solvers, and batched
GPU solvers. The ADMM, adapter, PyGRANSO integration, tests, and evidence schema
must not change for a solver replacement.
