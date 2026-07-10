# Phase 3.1 Implementation Plan: Builtin OSQP Parity

## Goal

Implement **builtin OSQP parity for the Torch-direct adapter**.

This feature should allow the system to:

1. Compare Torch and builtin OSQP under identical adapter settings.
2. Report status compatibility, feasibility, stationarity, residuals, and normalized objective gaps.
3. Avoid comparing iterates or iteration counts as correctness requirements.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

---

## Current State

The project already has:

* A builtin OSQP path for CPU solves.
* A Torch OSQP path for dense reference solves.
* Documentation requiring observable agreement with builtin OSQP.

The missing part is:

* A shared parity metric layer used by tests and fallback telemetry.
* A builtin-result adapter that normalizes info fields to match Torch diagnostics.
* Differential tests that use identical settings for both backends.

---

## New Components to Add

### Component 1

```text
CommonOSQPSettings
```

Responsibility:

```text
Represent the settings that must be applied identically to builtin and Torch OSQP paths.
```

### Component 2

```text
CommonAdapterMetrics
```

Responsibility:

```text
Compute backend-neutral feasibility, stationarity, residual, and objective-gap metrics.
```

### Component 3

```text
DifferentialComparison
```

Responsibility:

```text
Summarize compatibility between builtin and Torch results without requiring identical iterates.
```

---

## Class / Registry Diagrams

### Diagram 1: Common Settings

```text
+-------------------------------------------------------------------------------+
|                                CommonOSQPSettings                              |
+-------------------------------------------------------------------------------+
|  - rho: float = 0.1                                                            |
|  - sigma: float = 1e-6                                                         |
|  - alpha: float = 1.6                                                          |
|  - max_iter: int = 4000                                                        |
|  - check_termination: int = 25                                                 |
|  - scaling: int = 10                                                           |
|  - adaptive_rho: bool = true                                                   |
|  - adaptive_rho_interval: int = 50                                             |
|  - adaptive_rho_tolerance: float = 5                                           |
+-------------------------------------------------------------------------------+
|  + for_builtin(): dict                  --> OSQP Python settings              |
|  + for_torch(): TorchOSQPSettings       --> Torch settings                    |
|  + tolerances(dtype): ToleranceSpec      --> float64/float32 tolerances        |
+-------------------------------------------------------------------------------+
```

### Diagram 2: Common Adapter Metrics

```text
+-------------------------------------------------------------------------------+
|                              CommonAdapterMetrics                              |
+-------------------------------------------------------------------------------+
|  - toleranceSpec: ToleranceSpec                                                |
+-------------------------------------------------------------------------------+
|  + feasibility(A, l, u, x): float       --> bound violation                   |
|  + stationarity(P, q, A, x, y): float   --> KKT stationarity residual          |
|  + objective(P, q, x): float            --> primal objective                  |
|  + objective_gap(torch, builtin): float  --> normalized objective gap          |
+-------------------------------------------------------------------------------+
```

### Diagram 3: Differential Comparison

```text
+-------------------------------------------------------------------------------+
|                              DifferentialComparison                            |
+-------------------------------------------------------------------------------+
|  - torchResult: dict                                                           |
|  - builtinResult: dict                                                         |
|  - metrics: CommonAdapterMetrics                                               |
+-------------------------------------------------------------------------------+
|  + status_compatible(): boolean         --> solved/limited/error class check   |
|  + residuals_compatible(): boolean      --> residual thresholds                |
|  + objective_compatible(): boolean      --> normalized objective gap check     |
|  + report(): dict                       --> case-level comparison output       |
+-------------------------------------------------------------------------------+
```

---

## Class Diagram Rules

* Parity checks compare mathematical outcomes, not solver internals.
* The builtin path remains the CPU fallback and reference oracle for supported cases.
* Metric code should be reusable by unit tests, differential tests, and stability evidence generation.
* The metric layer must not mutate solver state.

---

## Data Model

```text
ToleranceSpec
  dtype: torch.dtype
  feasibility_tol: float
  stationarity_tol: float
  residual_tol: float
  objective_gap_tol: float
```

```text
BackendResultView
  backend: string
  status: string
  x: array or Tensor
  y: array or Tensor
  objective: float
  pri_res: float
  dua_res: float
  info: dict
```

```text
ComparisonReport
  case_id: string
  status_compatible: bool
  feasibility_ok: bool
  stationarity_ok: bool
  residuals_ok: bool
  objective_gap_ok: bool
  notes: list[string]
```

---

## Storage / State

* Store no persistent parity state in the solver.
* Store case-level reports only in test artifacts or stability evidence outputs.
* Use workspace diagnostics as read-only input for comparison reports.

---

## Required Methods

* `_common_osqp_settings(options, dtype)`.
* `_solve_builtin_osqp_common(input, settings)`.
* `_backend_result_view(result)`.
* `_compute_common_metrics(P, q, A, l, u, result)`.
* `_compare_backend_results(torch_result, builtin_result, tolerances)`.
* `_normalized_objective_gap(obj_a, obj_b)`.

---

## Validation Rules

* Differential tests must use identical settings for Torch and builtin OSQP.
* Float64 tolerance defaults to `1e-8`.
* Float32 tolerance defaults to `1e-5`.
* Status compatibility should allow equivalent solved classes but reject hidden numerical failure.
* Do not require identical `x`, `y`, `z`, iteration counts, or rho histories.
* Objective gaps must be normalized to avoid false failures near large objectives.

---

## UI / API Integration

* Keep parity metrics internal to tests, diagnostics, and telemetry.
* Include comparison fields in stability CSV and Markdown summaries.
* Do not expose a new public comparison API unless a later milestone requires it.

---

## Workflow

1. Build a supported dense QP case.
2. Convert options into `CommonOSQPSettings`.
3. Solve once with builtin OSQP.
4. Solve once with Torch OSQP.
5. Convert both results into `BackendResultView`.
6. Compute feasibility, stationarity, residual, and objective metrics.
7. Write a `ComparisonReport`.
8. Fail the gate only on unexplained unsupported status, residual, feasibility, stationarity, or objective-gap failure.

---

## Files to Create

* `tests/test_torch_osqp_builtin_parity.py`: deterministic parity cases and result comparison tests.
* `tests/helpers/torch_osqp_parity.py`: optional shared comparison helpers if test reuse becomes large.

---

## Files to Modify

* `pygranso/private/solveQP.py`: share common settings and result views where needed.
* `pygranso/private/torchOSQP.py`: expose internal diagnostics required for parity reports.
* `docs/TORCH_OSQP_COMPLETION_AUDIT.md`: record parity status.

---

## Error Handling

* Treat builtin OSQP setup errors as test setup failures.
* Treat Torch numerical failures as backend failures, not infeasibility.
* Include both backend info dictionaries in differential failure messages.
* Serialize failing QP inputs for stability reproduction in later phases.

---

## Testing Checklist

- [x] Identical settings are passed to builtin and Torch paths.
- [x] Float64 parity uses `1e-8` tolerances.
- [x] Float32 parity uses `1e-5` tolerances.
- [x] Status compatibility is checked independently from iterates.
- [x] Feasibility and stationarity metrics catch intentionally corrupted results.
- [x] Normalized objective gap handles near-zero and large objectives.
- [x] Differential failure output includes enough telemetry to reproduce the case.

---

## Acceptance Criteria

* Builtin and Torch results have a shared comparison vocabulary.
* Differential tests check mathematical agreement, not implementation identity.
* Parity evidence can be reused by the stability evidence package.
