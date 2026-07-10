# Phase 2.3 Implementation Plan: Scaling, Adaptive Rho, and Polishing

## Goal

Implement **Ruiz scaling, adaptive `rho`, strict polishing, and structurally compatible warm starts**.

This feature should allow the system to:

1. Preserve the mathematically correct scaling, residual, adaptive-`rho`, and polishing behavior from the revised pipeline.
2. Keep all scaling and warm-state data private to the optimizer-owned workspace.
3. Raise on requested polishing failure instead of hiding it behind a successful status.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

---

## Current State

The project already has:

* The dense LU boundary and workspace lifecycle planned in Phase 2.1.
* The direct ADMM loop planned in Phase 2.2.
* Documentation requiring Ruiz scaling, adaptive `rho`, polishing, and warm starts.

The missing part is:

* A clear data model for scaling maps and warm-state transformations.
* Deterministic adaptive-`rho` behavior shared with builtin OSQP settings.
* Strict polishing behavior and tests for both success and failure.

---

## New Components to Add

### Component 1

```text
RuizScalingCache
```

Responsibility:

```text
Store diagonal problem scaling, objective scaling, and unscale operations for dense Torch QPs.
```

### Component 2

```text
AdaptiveRhoController
```

Responsibility:

```text
Update rho deterministically based on residual balance and request workspace refactorization when rho changes.
```

### Component 3

```text
PolishingSolver
```

Responsibility:

```text
Perform the optional active-set polishing solve and enforce requested-polishing failure semantics.
```

### Component 4

```text
WarmStartMapper
```

Responsibility:

```text
Preserve, scale, unscale, and invalidate warm x/z/y state only for structurally compatible problems.
```

---

## Class / Registry Diagrams

### Diagram 1: Ruiz Scaling Cache

```text
+-------------------------------------------------------------------------------+
|                                RuizScalingCache                                |
+-------------------------------------------------------------------------------+
|  - D: Tensor[n,1]                                                              |
|  - E: Tensor[m,1]                                                              |
|  - c: Tensor[1]                                                                |
|  - passes: int                                                                 |
+-------------------------------------------------------------------------------+
|  + fit(P, q, A, l, u): ScaledQP          --> builds scaled tensors             |
|  + scale_state(x, z, y): ADMMState       --> maps warm state into scaled space |
|  + unscale_solution(x, y): Solution      --> maps solver output to user space  |
|  + compatible(signature): bool           --> checks reuse eligibility          |
+-------------------------------------------------------------------------------+
```

### Diagram 2: Adaptive Rho Controller

```text
+-------------------------------------------------------------------------------+
|                              AdaptiveRhoController                             |
+-------------------------------------------------------------------------------+
|  - interval: int                                                               |
|  - tolerance: float                                                            |
|  - enabled: bool                                                               |
+-------------------------------------------------------------------------------+
|  + should_check(iter): boolean          --> deterministic interval gate        |
|  + propose(pri_res, dua_res, rho): rho  --> residual-balance update           |
|  + apply(workspace, rho): None          --> invalidates factorization on change|
+-------------------------------------------------------------------------------+
```

### Diagram 3: Polishing Solver

```text
+-------------------------------------------------------------------------------+
|                                  PolishingSolver                               |
+-------------------------------------------------------------------------------+
|  - requested: bool                                                             |
|  - regularization: float                                                       |
+-------------------------------------------------------------------------------+
|  + identify_active_set(z, l, u): ActiveSet  --> bound activity mask            |
|  + solve_reduced_kkt(input, activeSet): Sol  --> dense correction solve        |
|  + accept(candidate, base): Solution         --> residual/objective gate       |
|  + require_success(result): None             --> raises if requested failed    |
+-------------------------------------------------------------------------------+
```

### Diagram 4: Warm Start Mapper

```text
+-------------------------------------------------------------------------------+
|                                  WarmStartMapper                               |
+-------------------------------------------------------------------------------+
|  - workspace: TorchOSQPWorkspace                                               |
+-------------------------------------------------------------------------------+
|  + can_reuse(signature): boolean          --> structure/order compatibility    |
|  + load(defaults): ADMMState              --> restores x/z/y or cold starts    |
|  + store(finalState): None                --> persists final x/z/y             |
|  + invalidate(reason): None               --> clears incompatible warm state   |
+-------------------------------------------------------------------------------+
```

---

## Class Diagram Rules

* Scaling, adaptive `rho`, polishing, and warm starts remain private implementation details.
* Workspace invalidation must be explicit and reason-coded.
* Adaptive `rho` may change numeric factorization state but must not change public backend selection.
* Polishing uses dense linear algebra only in this milestone.

---

## Data Model

```text
ScaledQP
  P_scaled: Tensor[n,n]
  q_scaled: Tensor[n,1]
  A_scaled: Tensor[m,n]
  l_scaled: Tensor[m,1]
  u_scaled: Tensor[m,1]
  scaling: RuizScalingCache
```

```text
AdaptiveRhoState
  rho: Tensor or scalar
  last_update_iter: int
  refactorization_required: bool
  update_count: int
```

```text
PolishResult
  attempted: bool
  success: bool
  reason: string
  x: Tensor[n,1]
  y: Tensor[m,1]
  objective: float
  residuals: dict
```

---

## Storage / State

* Store scaling cache, adaptive-`rho` state, and warm vectors in `TorchOSQPWorkspace`.
* Store polishing active sets and reduced solves as local temporaries.
* Invalidate scaling and warm starts on dimension, order, structure, dtype, device, or backend changes.
* Preserve `x`, `z`, and `y` for compatible matrix-value updates.
* Refactorize when adaptive `rho` changes the KKT system.

---

## Required Methods

* `_ruiz_scale_problem(P, q, A, l, u, passes=10)`.
* `_ruiz_unscale_solution(x, y, scaling)`.
* `_maybe_update_rho(state, residuals, settings, workspace)`.
* `_invalidate_after_rho_change(workspace, new_rho)`.
* `_identify_polish_active_set(z, l, u)`.
* `_polish_solution(input, state, scaling)`.
* `_load_warm_start(workspace, signature)`.
* `_store_warm_start(workspace, state, signature)`.

---

## Validation Rules

* Scaling must preserve finite user data and allow infinite bounds.
* Scaling passes default to 10 and must be deterministic.
* Adaptive `rho` defaults to enabled, interval 50, tolerance 5.
* Adaptive `rho` must refactorize only when the effective KKT matrix changes.
* Polishing failure raises when polishing was requested.
* Warm starts must never cross incompatible dimensions, constraint order, dtype, device, backend, or structure.
* Residuals used for adaptive `rho` must be computed in the correct scaled or unscaled space by design, not by accident.

---

## UI / API Integration

* Use common adapter defaults for builtin and Torch:
  * `rho=0.1`
  * `sigma=1e-6`
  * `alpha=1.6`
  * `max_iter=4000`
  * `check_termination=25`
  * Ruiz scaling passes `10`
  * adaptive `rho` interval `50`
  * adaptive `rho` tolerance `5`
* Expose polishing and warm-start behavior through existing options only.
* Do not add a public linear-solver selector.

---

## Workflow

1. Validate the original dense QP.
2. Build or reuse a compatible `RuizScalingCache`.
3. Map warm state into scaled solver space when allowed.
4. Run ADMM with deterministic adaptive-`rho` checks.
5. Refactorize through the workspace when `rho` changes.
6. Attempt polishing when requested.
7. If polishing succeeds, return the polished solution.
8. If polishing is requested and fails, raise with actionable diagnostics.
9. Unscale the final solution and persist compatible warm state.

---

## Files to Create

* `tests/test_torch_osqp_scaling_adaptive_polishing.py`: scaling, adaptive-`rho`, polishing, and warm-start tests.

---

## Files to Modify

* `pygranso/private/torchOSQP.py`: add scaling, adaptive-`rho`, polishing, and warm-state hooks.
* `pygranso/private/osqpWorkspace.py`: store scaling cache, adaptive-`rho` state, and invalidation reasons.
* `docs/TORCH_OSQP_COMPLETION_AUDIT.md`: track implementation and test evidence.

---

## Error Handling

* Raise `ValueError` for invalid scaling inputs.
* Raise a Torch OSQP numerical error for nonfinite scaled tensors or residuals.
* Raise a polishing-specific error when requested polishing fails.
* Attach workspace invalidation reason and adaptive-`rho` update count to diagnostics.

---

## Testing Checklist

- [x] Ruiz scaling is deterministic for fixed inputs.
- [x] Scaled/unscaled solutions preserve feasibility and objective within tolerance.
- [x] Infinite bounds remain valid through scaling.
- [x] Adaptive `rho` updates at deterministic intervals only.
- [x] Adaptive `rho` triggers refactorization exactly when needed.
- [x] Warm starts are reused for compatible value updates.
- [x] Warm starts are invalidated for structure, dtype, device, order, and backend changes.
- [x] Requested polishing success improves or preserves accepted residual/objective diagnostics.
- [x] Requested polishing failure raises.

---

## Acceptance Criteria

* Torch and builtin paths share the documented default settings.
* Scaling, adaptive `rho`, polishing, and warm starts are private, deterministic, and test-covered.
* Polishing failures are visible when polishing is requested.
* Workspace state remains local to a BFGS-SQP run and cannot leak through module globals.
