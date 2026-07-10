# PyGRANSO Torch-OSQP Full Development and Validation Pipeline Roadmap

Project Name: PyGRANSO Torch-OSQP Dense Reference Route
Version: 2.2
Date: 2026-07-04
Status: Release-candidate roadmap; dense CPU path implemented/evidenced, accelerator promotion still gated

This roadmap applies the full development and validation template to the current
PyGRANSO Torch-OSQP project. It uses the repository's existing phase files and
does not create new phase numbering unless the implementation scope changes.

The detailed phase plans are linked from the milestone roadmap in Section 20.

---

## Part I - Executive Summary

### 1. Decision

Build a correctness-first, Torch-native dense OSQP reference route for PyGRANSO.

This implementation is:

* correctness-first;
* PyTorch-native;
* dense reference, not sparse/scalable;
* not claimed as primal-infeasibility, dual-infeasibility, nonconvex-detection, or sparse large-scale support yet;
* designed so future Torch/sparse/vendor backends can fit behind the same internal factorize/solve interface.

The first trustworthy milestone supports feasible convex QPs in OSQP form:

```text
minimize    0.5 * x' P x + q' x
subject to  l <= A x <= u
```

The implementation is judged by observable outcomes:

* compatible status;
* feasibility;
* stationarity;
* primal/dual residuals;
* normalized objective gap;
* structured backend, fallback, linear-solve, and stability diagnostics.

It does not require identical ADMM trajectories, raw iterates, iteration counts,
rho histories, or low-level runtime behavior across devices/backends.

### 2. Primary Acceptance Envelope

| Item | Supported Contract |
| --- | --- |
| Problem class | Feasible convex QPs in OSQP form |
| Authoritative precision | float64 |
| Qualified precision | float32 with `1e-5` tolerances and observed conditioning around `1e2` |
| Size limit | Dense KKT dimension `n + m <= 2400` for automatic Torch selection |
| Float64 supported conditioning | Estimated KKT conditioning approximately `1e8` |
| Stress-only conditioning | Estimated KKT conditioning approaching `1e10` |
| Linear algebra | Reusable `torch.linalg.lu_factor_ex` and `torch.linalg.lu_solve` |
| Required numerical features | Ruiz scaling, adaptive rho, polishing, warm starts |
| Default CPU policy | Builtin OSQP |
| Default accelerator policy | Torch only after backend-specific promotion gates pass |
| Failure budget | Zero unexplained failures inside the supported size/dtype/conditioning/device matrix |

### 3. Backend / Platform Support Matrix

| Backend / Platform | Precision | Current Status | Promotion Evidence Required |
| --- | --- | --- | --- |
| Torch CPU on Linux | float32/float64 | Release-gated; core workflow evidence exists | Core, nightly, and end-to-end release evidence |
| Torch CPU on Windows | float32/float64 | Local manifest-backed correctness evidence exists | Release review of local/CI evidence |
| Torch CPU on macOS | float32/float64 | Release-gated; core workflow evidence exists | Core, nightly, and end-to-end release evidence |
| NVIDIA CUDA | float32/float64 | Unpromoted | Current-source real hardware correctness plus median end-to-end runtime no worse than 5x builtin CPU OSQP |
| AMD ROCm | float32/float64 | Unclaimed | Real-hardware runner and full evidence package |
| Apple MPS | float32 only | Unclaimed | Real Apple hardware reusable-LU evidence |
| Apple MPS float64 | Unsupported for Torch tensors | `auto` returns builtin CPU result | No Torch support claim |
| Sparse/direct/vendor backends | Not in first milestone | Future work | Same internal interface plus backend-specific evidence |

### 4. Outcome Policy

The default `auto` policy follows the requested optimization device:

* CPU work uses builtin OSQP.
* Accelerator work uses Torch only when the backend is promoted, the KKT size
  is inside the envelope, memory preflight passes, correctness evidence exists,
  and the performance sanity gate passes.
* Unsupported, unpromoted, oversized, memory-risky, failed, or unsolved Torch
  routes under `auto` produce warned builtin fallback with causal telemetry.
* Explicit `torch` requests never silently change backend. Above the supported
  size envelope they warn and attempt as requested; hard failures propagate to
  PyGRANSO's existing outer fallback strategy.

Automatic fallback must be:

* visible;
* warned;
* structured;
* reproducible;
* stored in diagnostics.

### 5. Main Risks and Controls

| Risk | Control |
| --- | --- |
| Repeated dense refactorization | Cache LU factors and rebuild only when KKT-affecting data changes |
| Dense memory growth | `n + m <= 2400` auto limit plus conservative memory preflight |
| Precision instability | Float64 authoritative path; dtype-aware tolerances; condition-aware float32 claims |
| Hidden fallback | Runtime warning plus structured telemetry |
| Cross-run contamination | One private `TorchOSQPWorkspace` per BFGS-SQP run |
| Misclassified numerical failure | Never report LU/numerical failure as infeasibility |
| Optional polishing corrupts solution | Accept only nonworse/tolerance-satisfying candidates; requested failure raises |
| Unsupported accelerator claims | Promote backend-by-backend only after evidence |
| Workflow unavailable before merge | Keep feature-branch workflows configured but not claimed until registered on default branch |

---

## Part II - Decision-Complete Engineering Specification

### 6. Architecture

```text
PyGRANSO BFGS-SQP
  -> steering QP or stationarity QP
  -> pygranso/private/solveQP.py
  -> pygranso/private/osqpTorchAdapter.py
  -> canonical P, q, A_osqp, l, u
     -> builtin CPU OSQP
     -> dense Torch OSQP reference
        -> OSQP ADMM equations
        -> private DenseLUSolver boundary
        -> reusable PyTorch LU factorization and repeated solves
```

The low-level solver boundary is internal.

User-facing backend selector:

```text
osqp_algebra = {auto, builtin, torch}
```

Do not expose nested low-level solver selectors until there are at least two
validated implementations behind the same internal boundary.

### 7. Public Contract

Goal: define public behavior before changing solver internals.

Inputs:

* `H`: PyGRANSO QP Hessian-like matrix.
* `f`: PyGRANSO QP linear vector.
* `A`, `b`: inequality constraint matrix/vector.
* `LB`, `UB`: variable lower/upper bounds.
* `torch_device`: requested optimization device.
* `double_precision`: selects float64 or float32 behavior.
* `osqp_options`: contains `algebra`, common settings, and optional workspace.

Outputs:

* canonical OSQP problem `P`, `q`, `A_osqp`, `l`, `u`;
* backend selection result;
* solution/result tensor;
* result on the requested or documented fallback device;
* structured diagnostics when requested.

Public settings:

* accepted backend values: `auto`, `builtin`, `torch`;
* accepted precisions: float32 and float64;
* legacy CG/CUDA Graph settings: actionable migration error path;
* unsupported explicit operations: clear failure rather than silent fallback.

Validation:

* Accepted `osqp_algebra` values are exactly `auto`, `builtin`, and `torch`.
* CPU `auto` uses builtin OSQP.
* Unsupported/unpromoted accelerator `auto` falls back with warning and telemetry.
* Explicit `torch` never silently changes backend.
* Unsupported options raise clear errors before low-level kernels.

Exit Criteria:

* Public behavior is stable.
* Backend selection is deterministic and testable.
* Fallback behavior is visible and recorded.
* Invalid inputs fail before reaching low-level linear algebra.

### 8. Canonical Problem Contract

The canonical QP is:

```text
minimize    0.5 * x' P x + q' x
subject to  l <= A x <= u
```

Requirements:

* `P` is dense, finite, square, and symmetric within dtype-aware tolerance.
* `q` is finite and compatible with `P`.
* `A` is finite and has compatible row/column dimensions.
* `l` and `u` may contain allowed infinities but never NaN.
* `l <= u`.
* All Torch tensors use the same device and dtype after normalization.
* Only float32 and float64 are accepted.
* Near-symmetric `P` may be symmetrized only within a tolerance proportional to
  machine epsilon and `max(1, ||P||_inf)`.
* Expensive eigenvalue/conditioning diagnostics are available for tests,
  evidence, and debugging but are not always paid in normal solves.

Validation Checklist:

* Reject invalid shapes.
* Reject NaN and unsupported Inf.
* Reject unsupported dtype.
* Normalize device and dtype consistently.
* Preserve documented fallback behavior.
* Add diagnostic checks for tests and evidence runners.

### 9. Core Class / Registry Diagrams

Use one ASCII diagram for every major class/module. The names below are the
current project names; implementation plans may add helper names only when the
source code actually introduces them.

#### 9.1 Workspace Class

```text
+-------------------------------------------------------------------------------+
|                              TorchOSQPWorkspace                                |
+-------------------------------------------------------------------------------+
|  - state: Dict[str, Tensor]                                                    |
|  - problem_signature: Tuple | null                                             |
|  - constraint_order_signature: Tuple | null                                    |
|  - p_pattern: Tuple | null                                                     |
|  - a_pattern: Tuple | null                                                     |
|  - scaling: Dict[str, Tensor] | null                                           |
|  - rho_bar: Tensor | null                                                      |
|  - rho_setting: float | null                                                   |
|  - active_backend: string | null                                               |
|  - builtin_cache: Any | null                                                   |
|  - builtin_stats: Dict[str, int]                                               |
|  - linear_solver: DenseLUSolver                                                |
|  - last_info: Dict[str, Any] | null                                            |
+-------------------------------------------------------------------------------+
|  + ensure_backend(backend): bool        --> Switches backend and resets safely |
|  + reset_torch(): void                  --> Clears Torch reusable state        |
|  + reset_all(): void                    --> Clears all cached state            |
|  + update_state(x, z, y): void          --> Saves warm-start state             |
|  + set_scaling(...): void               --> Stores scaling metadata            |
|  + set_diagnostics(info): void          --> Stores latest diagnostics          |
|  + clear_factors(): void                --> Clears cached factorization        |
+-------------------------------------------------------------------------------+
```

Ownership rule: one BFGS-SQP run owns one private workspace. No module global may
store warm state, solver factors, or backend cache.

#### 9.2 Dense Linear Solver Class

```text
+-------------------------------------------------------------------------------+
|                                DenseLUSolver                                   |
+-------------------------------------------------------------------------------+
|  - matrix: Tensor | null                                                       |
|  - lu: Tensor | null                                                           |
|  - pivots: Tensor | null                                                       |
|  - factorization_status: Tensor | null                                         |
|  - factorization_count: int                                                    |
|  - solve_count: int                                                            |
+-------------------------------------------------------------------------------+
|  + factorize(K): void                   --> Validates and factorizes matrix    |
|  + solve(rhs): Tensor                   --> Solves using cached factors        |
|  + factorize_if_needed(K): bool         --> Reuses or rebuilds factorization   |
|  + clear(): void                        --> Clears matrix and factors          |
|  + diagnostics(): LinearSolveDiagnostics --> Returns counters/status           |
+-------------------------------------------------------------------------------+
```

Validation rule: reject non-square matrices, nonfinite values, unsupported
dtypes, failed factorization info, nonfinite factors, invalid RHS, and nonfinite
solutions.

#### 9.3 Linear Solve Diagnostics

```text
+-------------------------------------------------------------------------------+
|                           LinearSolveDiagnostics                               |
+-------------------------------------------------------------------------------+
|  - solver_name: string                                                         |
|  - factorization_info: int | string | null                                     |
|  - factorization_count: int                                                    |
|  - solve_count: int                                                            |
|  - absolute_residual: float | null                                             |
|  - relative_residual: float | null                                             |
+-------------------------------------------------------------------------------+
|  + to_dict(): Dict[str, Any]            --> Serializes diagnostics             |
|  + is_success(): bool                   --> Reports factor/solve success       |
+-------------------------------------------------------------------------------+
```

#### 9.4 Backend Selection / Adapter Module

```text
+-------------------------------------------------------------------------------+
|                         pygranso/private/osqpTorchAdapter.py                   |
+-------------------------------------------------------------------------------+
|  - DEFAULT_OSQP_SETTINGS: Dict[str, Any]                                       |
|  - PROMOTED_ACCELERATOR_BACKENDS: Dict[str, bool]                              |
|  - MAX_SUPPORTED_KKT_DIM: int                                                  |
|  - LINEAR_SOLVER_OPTION_KEYS: Set[str]                                         |
+-------------------------------------------------------------------------------+
|  + solve_osqp_torch_qp(...): Result     --> Public adapter solve entry point   |
|  + _select_backend(...): Dict           --> Applies auto/builtin/torch policy  |
|  + estimate_dense_kkt(...): Tuple       --> Estimates KKT size and memory      |
|  + _solve_builtin_osqp_path(...): Result--> Calls trusted builtin backend      |
|  + _solve_torch_osqp_path(...): Result  --> Calls dense Torch implementation   |
|  + _selection_fallback(...): Dict       --> Builds visible fallback metadata   |
+-------------------------------------------------------------------------------+
```

#### 9.5 Direct Algorithm Kernel

```text
+-------------------------------------------------------------------------------+
|                         pygranso/private/torchOSQP.py                          |
+-------------------------------------------------------------------------------+
|  - No persistent global state                                                  |
+-------------------------------------------------------------------------------+
|  + build_kkt_matrix(...): Tensor        --> Builds dense KKT matrix            |
|  + build_kkt_rhs(...): Tensor           --> Builds solve right-hand side       |
|  + recover_z_tilde(...): Tensor         --> Recovers intermediate constraint   |
|  + admm_vector_update(...): Tuple       --> Performs ADMM vector update        |
|  + solve_torch_osqp_direct(...): Dict   --> Main direct ADMM loop              |
+-------------------------------------------------------------------------------+
```

Rule: the kernel receives all reusable state through arguments or workspace. It
does not own cross-run state.

#### 9.6 Scaling / Numerical Feature Module

```text
+-------------------------------------------------------------------------------+
|                    torchOSQP scaling/adaptive/polishing helpers                |
+-------------------------------------------------------------------------------+
|  - No persistent global state                                                  |
+-------------------------------------------------------------------------------+
|  + _scaling_for_problem(...): Dict      --> Computes or reuses Ruiz scaling    |
|  + _scale_problem(...): Tuple           --> Produces scaled QP tensors         |
|  + _initial_scaled_state(...): Tuple    --> Creates compatible initial state   |
|  + _unscale_state(...): Tuple           --> Converts result to original coords |
|  + _adaptive_rho_update(...): Tuple     --> Applies deterministic rho update   |
|  + _polish_solution(...): Dict          --> Runs optional dense polishing      |
|  + _residuals(...): Tuple               --> Computes KKT acceptance metrics    |
+-------------------------------------------------------------------------------+
```

#### 9.7 Evidence / Stability Runner

```text
+-------------------------------------------------------------------------------+
|                               torch_osqp_stability.py                          |
+-------------------------------------------------------------------------------+
|  - seed_list: List[int]                                                        |
|  - stress_seed_list: List[int]                                                 |
|  - backend/device/dtype matrix                                                 |
|  - output_dir: Path                                                            |
|  - time_limit_seconds: float                                                   |
+-------------------------------------------------------------------------------+
|  + run_case(...): Dict                  --> Runs one deterministic case        |
|  + write_results_csv(...): Path         --> Writes case-level results          |
|  + write_manifest(...): Path            --> Writes provenance/settings data    |
|  + write_summary(...): Path             --> Writes human-readable summary      |
|  + save_failure_case(...): Path         --> Saves reproduction artifact        |
+-------------------------------------------------------------------------------+
```

### 10. Workspace Lifecycle Contract

| Operation | Input Change | Preserved Output | Invalidated Output |
| --- | --- | --- | --- |
| Vector-only update | Same `P`/`A` values and structure, new `q`/`l`/`u` | `x`, `z`, `y`, rho, scaling, LU | none |
| Matrix-value update | Same shape/pattern/dtype/device/order, changed `P` or `A` values | `x`, `z`, `y` | scaling and LU |
| Parameter change | Compatible problem, changed rho or sigma | `x`, `z`, `y` | LU |
| Structure change | Changed shape, pattern, or constraint ordering | none | warm state, scaling, LU |
| Dtype/device/backend change | Changed dtype, device, or selected backend | none | complete workspace |

Workspace validation:

* Unit-test vector-only update reuse.
* Unit-test matrix-value update refactorization.
* Unit-test parameter-change refactorization.
* Unit-test structure-change invalidation.
* Unit-test dtype/device/backend complete reset.
* Confirm diagnostics explain reuse, rebuild, and reset.

Exit Criteria:

* No global warm state exists.
* Every reusable object has one owner.
* Every invalidation path is deterministic.
* Reuse is visible through diagnostics.

### 11. Dense Linear-Solver Lifecycle

```text
factorize(K)
  validate square shape, dtype, device, and finite values
  call torch.linalg.lu_factor_ex(K, check_errors=False)
  reject failed factorization info
  reject nonfinite factors
  cache matrix clone, factors, pivots/status
  increment factorization_count

solve(rhs)
  normalize vector RHS to shape (n, 1)
  validate shape, dtype, device, and finite values
  call torch.linalg.lu_solve(LU, pivots, rhs)
  reject nonfinite solution
  optionally report ||Kx-b|| / max(1, ||b||)
  increment solve_count

factorize_if_needed(K)
  compare K to cached matrix using documented compatibility rule
  reuse if compatible
  refactorize if incompatible
```

Validation:

* Reject non-square matrix.
* Reject nonfinite matrix.
* Reject unsupported dtype.
* Reject RHS with wrong shape.
* Support vector RHS and matrix RHS.
* Restore vector output shape when appropriate.
* Detect failed factorization.
* Detect nonfinite solution.
* Report factorization and solve counters.

### 12. Algorithm Equations / Core Math

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
z_next = project_box(z_relaxed + y/rho, l, u)
y_next = y + rho * (z_relaxed - z_next)
```

Acceptance metrics are evaluated in original coordinates:

```text
r_primal = ||A x - z||_inf
r_dual   = ||P x + q + A' y||_inf
objective = 0.5 * x' P x + q' x
```

Rules:

* Preserve the OSQP ADMM equations.
* Do not change equations to make tests pass.
* Evaluate final residuals/metrics in original coordinates.
* Accept nonunique solutions by observable metrics, not exact vector equality.
* Raise or report numerical failure as numerical failure, not infeasibility.

### 13. Required Numerical Features

#### 13.1 Scaling

Use ten deterministic Ruiz diagonal-equilibration passes.

Rules:

* Compute scaling deterministically.
* Cache scaling for compatible vector-only updates.
* Solve the scaled problem if enabled.
* Unscale `x`, `z`, and `y` before reporting.
* Report acceptance metrics in original coordinates.

#### 13.2 Adaptive Parameter Update

Use deterministic adaptive rho with interval `50` and tolerance `5`.

Rules:

* Update rho deterministically.
* Apply the larger vector-valued rho policy for equality rows.
* Rebuild factorization after an accepted rho change.
* Continue from current `x`, `z`, and `y`.
* Record rho changes in diagnostics.

#### 13.3 Polishing / Refinement

Use dense active-set polishing through the same LU boundary.

Rules:

* Build polishing systems through `DenseLUSolver`.
* Reuse factorization when valid.
* Accept a candidate only when its KKT metric is nonworse or satisfies tolerance.
* Raise requested polishing failure when polishing was requested and cannot be accepted.
* Never silently degrade an accepted result.

#### 13.4 Warm Starts

Rules:

* Enable warm starts internally.
* Reuse warm state for compatible updates.
* Recompute scaling and factors when matrix values change.
* Clear workspace on structural, dtype, device, or backend changes.
* Make warm-start reuse observable through diagnostics.

### 14. Defaults

| Setting | Default |
| --- | --- |
| `rho` | `0.1` |
| `sigma` | `1e-6` |
| `alpha` | `1.6` |
| `max_iter` | `4000` |
| `check_termination` | `25` |
| float64 `eps_abs` | `1e-8` |
| float64 `eps_rel` | `1e-8` |
| float32 `eps_abs` | `1e-5` |
| float32 `eps_rel` | `1e-5` |
| scaling | `10` Ruiz passes |
| adaptive rho | `true` |
| rho update interval | `50` |
| rho update tolerance | `5` |
| polishing | `true` |
| warm start | `true` |

### 15. Backend Selection and Fallback

| Request | Behavior |
| --- | --- |
| `auto` on CPU | Use builtin OSQP |
| `auto` on validated accelerator inside limits | Use Torch backend |
| `auto` on unsupported/unpromoted accelerator | Warn and use builtin OSQP |
| `auto` above size or memory envelope | Warn and use builtin OSQP |
| `auto` Torch exception | Warn, retry builtin OSQP, retain causal telemetry |
| `auto` Torch unsolved status | Warn, retry builtin OSQP, retain causal telemetry |
| explicit `builtin` | Use builtin OSQP |
| explicit `torch` inside limits | Use Torch backend |
| explicit `torch` above limits | Warn and attempt; never silently change backend |

Fallback diagnostics must include:

* requested backend;
* selected backend;
* fallback backend;
* trigger;
* original exception or status;
* fallback status;
* device-transfer flag;
* result device;
* reproducibility notes.

### 16. Status and Error Semantics

Return structured statuses for:

* solved;
* maximum iterations;
* fallback success;
* fallback failure;
* unsupported auto route;
* stress-only classification.

Raise for:

* invalid inputs;
* unsupported explicit operations;
* failed factorization;
* NaN or Inf where forbidden;
* numerical polishing failure when polishing is requested;
* impossible workspace state;
* failed explicit backend request.

Do not claim yet:

* primal infeasibility certificate;
* dual infeasibility certificate;
* nonconvex detection;
* sparse large-scale performance;
* unsupported accelerator support.

Move these to future work unless implementation and tests exist.

### 17. Validation Pipeline

```text
Low-level solver tests
  -> KKT assembly and equation tests
  -> deterministic complete-problem tests
  -> scaling/adaptive-rho/polishing/warm-state tests
  -> builtin-vs-Torch differential tests
  -> PyGRANSO integration contracts
  -> metamorphic tests
  -> seeded randomized and conditioning tests
  -> end-to-end workloads
  -> backend-specific hardware gates
  -> performance sanity gate
```

Core validation:

* Linear solver tests.
* KKT/system matrix assembly tests.
* RHS construction tests.
* ADMM vector update equation tests.
* Projection/constraint handling tests.
* Finite solution tests.
* Residual and objective tests.

Differential validation:

* Compare builtin OSQP and Torch using identical settings.
* Compare status compatibility.
* Compare residuals.
* Compare objective gap.
* Compare feasibility and stationarity.
* Do not require exact iterate, trajectory, or iteration-count match.

Metamorphic validation:

* Scaling transformation preserves original-coordinate solution quality.
* Equivalent constraint/order cases preserve metrics or invalidate correctly.
* Vector-only updates reuse factors when allowed.
* Matrix changes refactorize when required.
* Dtype/device/backend changes reset state.

Randomized validation:

* Fixed seeds.
* Multiple size buckets.
* Multiple conditioning buckets.
* Multiple dtype buckets.
* Failure reproduction artifacts saved.
* Stress cases separated from supported claims.

End-to-end validation:

* Run PyGRANSO workloads.
* Validate steering, stationarity, penalty, and fallback contracts.
* Confirm fallback does not break caller contract.
* Confirm result device behavior.
* Confirm structured info is stable.

### 18. Evidence Package

Each stability or release-validation run must produce:

* `torch_osqp_stability_results.csv` with case-level gates;
* `torch_osqp_stability_manifest.json` with commit, platform, hardware, Python,
  PyTorch, OSQP, backend, settings, and seeds;
* `torch_osqp_stability_summary.md` with family totals and pass/fail classification;
* serialized reproduction file for every failure;
* CI artifacts for core, nightly, and hardware workflows when those workflows run.

Manifest requirements:

* commit hash;
* dirty-worktree state before generated artifacts;
* platform;
* hardware;
* Python version;
* PyTorch version;
* OSQP/library versions;
* backend;
* dtype;
* settings;
* seeds;
* time budget;
* `timed_out` flag;
* `partial_results` flag;
* last completed case;
* source hash over maintained code/tests/workflows/docs.

Timeout rule:

If the time budget is exceeded after a case completes:

* write all collected artifacts;
* mark `timed_out=true`;
* mark `partial_results=true`;
* record the last completed case;
* exit nonzero;
* classify the run as reproducible telemetry, not passing release evidence.

### 19. Performance Gate

Performance is not a correctness criterion. Performance controls only automatic
backend promotion.

Promotion rule:

```text
On representative workloads, Torch accelerator median end-to-end time must be
no worse than 5x builtin CPU OSQP.
```

If a backend passes correctness but fails performance:

* it remains explicit-only or unclaimed;
* `auto` falls back visibly;
* correctness evidence remains useful regression evidence.

Performance checklist:

* Benchmark includes setup and transfer cost.
* Benchmark uses representative workloads.
* Median runtime is reported.
* Slowdown ratio is reported.
* Backend promotion decision is recorded.
* Failing performance gate does not invalidate correctness evidence.

### 20. Milestone Roadmap

Each phase is complete only when:

* interface is implemented;
* data structures are implemented;
* tests are implemented;
* evidence artifacts are produced when required;
* exit criteria are satisfied;
* roadmap checkbox is updated only for verified work.

This repository keeps backend promotion in Phase 4.3 and documentation handoff
in Phase 5.1 to match existing plan files.

#### Phase 0 - Research Snapshot and Rollback Point

Goal: preserve prior experimental sparse-CG/CUDA Graph work before narrowing the
active package path.

Plan files:

* [Phase 0.1 archive snapshot](phase_0.1_archive_snapshot_plan.md)
* [Phase 0.2 remove research paths](phase_0.2_remove_research_paths_plan.md)

Tasks:

- [x] Create archive branch `archive/sparse-cg-cuda-graph`.
- [x] Create signed tag `research-sparse-cg-cuda-graph-final`.
- [x] Push archive branch and tag.
- [x] Record baseline validation result.
- [x] Remove unsupported research execution paths from active package code.
- [x] Verify active package no longer imports removed paths.

Exit Criteria:

* Reviewer can recover research snapshot from git.
* Active route contains no hidden unsupported sparse-CG/CUDA Graph execution path.

#### Phase 1 - Public Contract and Backend Policy

Goal: define public behavior before implementing/expanding dense backend internals.

Plan files:

* [Phase 1.1 public QP contract](phase_1.1_public_qp_contract_plan.md)
* [Phase 1.2 backend policy and fallback telemetry](phase_1.2_backend_policy_and_fallback_plan.md)
* [Phase 1.3 settings validation and migration](phase_1.3_settings_validation_and_migration_plan.md)

Tasks:

- [x] Define canonical input contract.
- [x] Define accepted backend options.
- [x] Define common default settings.
- [x] Implement backend selection.
- [x] Implement fallback telemetry shape.
- [x] Implement canonical problem builder.
- [x] Add unsupported-option migration errors.
- [x] Add public contract tests.

Exit Criteria:

* Public behavior is stable.
* Inner solver can evolve behind the adapter without changing API.

#### Phase 2.1 - Data Model, Workspace, and Factorization Lifecycle

Goal: make reusable state explicit, private, and safe across repeated solves.

Plan file:

* [Phase 2.1 dense LU workspace lifecycle](phase_2.1_dense_lu_workspace_plan.md)

Tasks:

- [x] Implement `TorchOSQPWorkspace`.
- [x] Implement `DenseLUSolver`.
- [x] Implement `LinearSolveDiagnostics`.
- [x] Implement workspace backend switching.
- [x] Implement factorization reuse.
- [x] Implement deterministic invalidation rules.
- [x] Add tests for vector-only updates.
- [x] Add tests for matrix-value updates.
- [x] Add tests for parameter updates.
- [x] Add tests for structure/dtype/device/backend changes.

Exit Criteria:

* No module global stores reusable state.
* Every reusable object has one owner.
* Diagnostics explain reused, rebuilt, and cleared factors.

#### Phase 2.2 - Dense Direct Algorithm Kernel

Goal: implement the dense reference ADMM algorithm around the factorization boundary.

Plan file:

* [Phase 2.2 direct ADMM kernel](phase_2.2_direct_admm_kernel_plan.md)

Tasks:

- [x] Build KKT/system matrix.
- [x] Build RHS.
- [x] Implement vector updates.
- [x] Implement projection/constraint update.
- [x] Implement residual computation.
- [x] Implement objective computation.
- [x] Implement solved/max-iteration status handling.
- [x] Update workspace state after solve.
- [x] Record factorization and iteration counters.

Exit Criteria:

* Algorithm equations match specification.
* Numerical failures are raised or reported as unsolved.
* Numerical failures are not mislabeled as infeasibility.

#### Phase 2.3 - Scaling, Adaptive Updates, Polishing, and Warm Starts

Goal: add numerical features required for observable agreement with builtin OSQP.

Plan file:

* [Phase 2.3 scaling, adaptive rho, polishing, and warm starts](phase_2.3_scaling_adaptive_polishing_plan.md)

Tasks:

- [x] Implement deterministic Ruiz scaling.
- [x] Implement problem scaling.
- [x] Implement state initialization.
- [x] Implement state unscaling.
- [x] Implement adaptive rho update.
- [x] Implement polishing/refinement.
- [x] Implement original-coordinate residuals.
- [x] Implement warm-start reuse.
- [x] Implement warm-start invalidation.

Exit Criteria:

* Acceptance metrics are reported in original coordinates.
* Polishing cannot silently degrade a valid result.
* Compatible warm starts are observable through diagnostics.

#### Phase 3 - Builtin Parity, Fallback, and PyGRANSO Integration

Goal: make builtin OSQP and dense Torch comparable through one adapter contract,
then verify the contract inside PyGRANSO.

Plan files:

* [Phase 3.1 builtin OSQP parity](phase_3.1_builtin_parity_plan.md)
* [Phase 3.2 PyGRANSO integration](phase_3.2_pygranso_integration_plan.md)

Tasks:

- [x] Implement builtin/reference backend path.
- [x] Implement shared metric computation.
- [x] Implement builtin cache/update behavior where compatible.
- [x] Implement fallback trigger handling.
- [x] Implement fallback telemetry.
- [x] Implement result device policy.
- [x] Implement memory and size preflight.
- [x] Add parity tests.
- [x] Add PyGRANSO steering/stationarity/fallback tests.

Exit Criteria:

* Automatic fallback is always visible.
* Explicit failures remain explicit.
* Backend comparison uses shared metrics.
* PyGRANSO's outer fallback contract remains intact.

#### Phase 4 - Validation Evidence and Backend Promotion

Goal: prove behavior through deterministic, differential, metamorphic,
randomized, end-to-end, and platform gates; promote backends only after evidence.

Plan files:

* [Phase 4.1 tests and differential validation](phase_4.1_tests_and_differential_plan.md)
* [Phase 4.2 stability evidence package](phase_4.2_stability_evidence_plan.md)
* [Phase 4.3 platform gates and backend promotion](phase_4.3_platform_promotion_plan.md)

Tasks:

- [x] Add deterministic unit test suite.
- [x] Add differential backend tests.
- [x] Add metamorphic tests.
- [x] Add randomized stability runner.
- [x] Add conditioning tests.
- [x] Add end-to-end workload/performance tests.
- [x] Add evidence CSV output.
- [x] Add evidence manifest output.
- [x] Add Markdown summary output.
- [x] Add failure reproduction output.
- [x] Add core CI workflow.
- [x] Add nightly workflow file on the feature branch.
- [x] Register and run nightly workflow on fork `main`.
- [x] Add hardware promotion workflow file on the feature branch.
- [ ] Register/merge nightly and hardware workflows on upstream `main` before relying on upstream schedules/dispatch.
- [ ] Promote CUDA only after representative correctness and <=5x performance evidence.
- [ ] Obtain ROCm and MPS runners before making either support claim.

Exit Criteria:

* Zero unexplained failures inside the supported matrix.
* Stress rows are classified as stress evidence, not support claims.
* Evidence package is reproducible.
* No backend is promoted by assumption.

#### Phase 5.1 - Documentation, PDF, and Release Handoff

Goal: make the implementation reviewable and reproducible.

Plan file:

* [Phase 5.1 documentation, PDF, and release handoff](phase_5.1_documentation_pdf_release_plan.md)

Tasks:

- [x] Maintain Markdown specification.
- [x] Render PDF.
- [x] Add completion audit.
- [x] Add code-edit log entries.
- [x] Verify docs match current source and evidence.
- [x] Verify evidence artifact expectations.
- [x] Verify roadmap checkboxes for already-implemented work.
- [x] Open or update release PR from `feature/torch-osqp-dense-reference`.
- [x] Verify fork `main` workflow registration and nightly CPU evidence.

Exit Criteria:

* Reviewer can follow public API, data model, solver lifecycle, validation, and
  release evidence without reading implementation code first.

#### Phase 5.2 - Fork-Local Release Candidate Hardening

Goal: treat the fork, not the original repo PR, as the active validation target, and review local/fork evidence.

Plan file:

* [phase_5.2_fork_local_hardening_plan.md](phase_5.2_fork_local_hardening_plan.md)

Tasks:

- [x] Run `git status --short --branch` to check local cleanliness.
- [x] Confirm latest fork `main` core and nightly workflows pass.
- [x] Aggregate and verify latest fork-main nightly artifacts (1800 CPU cases, 0 failures).
- [x] Run local deterministic tests with temp-directory bypass where necessary.
- [x] Document final release-candidate readiness status in external Report notebook.
- [x] Do not merge or trigger upstream pull request actions.

Exit Criteria:

* Fork-local release readiness is defined, verified, and documented.
* Core and nightly GHA workflows pass on the fork.
* No accelerator support is overclaimed.

### 21. Migration Sequence

1. [x] Preserve prior sparse-CG/CUDA Graph research snapshot.
2. [x] Create and push archive branch `archive/sparse-cg-cuda-graph`.
3. [x] Create and push signed tag `research-sparse-cg-cuda-graph-final`.
4. [x] Remove custom CG, Jacobi, sparse-operator, CUDA Graph, and selection code from the package path.
5. [x] Add workspace and linear-solver tests.
6. [x] Implement reusable factorization boundary.
7. [x] Refactor direct algorithm around that boundary.
8. [x] Add scaling, adaptive rho, polishing, and warm starts.
9. [x] Implement backend policy, migration errors, size guards, and fallback telemetry.
10. [x] Add differential, randomized, hardware-workflow, end-to-end, and reporting gates.
11. [ ] Promote each accelerator backend only after its correctness and performance evidence passes.
12. [x] Update release PR and verify fork default-branch workflow registration.
13. [ ] Obtain upstream review/merge and upstream default-branch workflow registration.
14. [x] After completed roadmap tasks, change only verified checkboxes from `[ ]` to `[x]`.

### 22. Decision Log

| Decision | Rationale |
| --- | --- |
| Dense reference route first | Correctness and maintainability are primary |
| Torch-native implementation | PyGRANSO already uses Torch tensors and device-aware optimization |
| Reusable factorization | Avoid repeated expensive setup inside ADMM and compatible updates |
| Observable agreement | Different devices/backends need not match trajectories |
| Float64 authoritative | Float32 degrades materially on ill-conditioned KKT systems |
| Float32 qualified separately | Audit evidence only supports looser/condition-limited float32 claims |
| Feasible convex QP scope first | Avoid unvalidated certificate and nonconvex claims |
| `auto` follows device only after gates | Avoid unsupported or unprofitable accelerator selection |
| Per-run workspace | Prevent cross-run state contamination |
| Backend-by-backend promotion | Support claims require real hardware and evidence |
| Five-times performance ceiling | Prevent severe automatic regressions without making speed a correctness criterion |
| Keep ROCm/MPS unclaimed | No real runner evidence exists yet |
| PR deferred until project perfect | PR #63 and workflow registration are deferred to prevent merging before all aspects of the solver and platform verification are perfect |
| Use built-in PyTorch routines | Avoid designing custom linear solvers; rely entirely on GPU-boosted built-in PyTorch linear algebra (e.g., lu_factor_ex/lu_solve) for correctness and standard acceleration |

### 23. Future Work

After the dense reference route passes applicable gates and release review, future
backends may implement the same internal interface.

Potential future backend candidates:

* sparse direct backend;
* iterative backend;
* GPU direct solver;
* batched solver;
* vendor-specific solver such as cuDSS;
* future PyTorch sparse solver;
* future Torch sparse linear algebra route.

Rules for future work:

* Do not change public API unnecessarily.
* Do not change evidence schema unnecessarily.
* Keep the same factorize/solve/refactorize contract.
* Keep existing tests.
* Add backend-specific evidence before promotion.
* Do not claim unsupported certificates or nonconvex detection until implemented and validated.
