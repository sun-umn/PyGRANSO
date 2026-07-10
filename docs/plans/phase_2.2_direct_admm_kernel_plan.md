# Phase 2.2 Implementation Plan: Direct ADMM Kernel

## Goal

Implement **the dense Torch-direct ADMM kernel**.

This feature should allow the system to:

1. Assemble the dense OSQP KKT system with the preserved KKT, projection, dual-update, and residual equations.
2. Run vector-only ADMM updates while reusing the workspace LU factorization from Phase 2.1.
3. Return OSQP-compatible status, residual, objective, and diagnostic fields without claiming infeasibility certificates.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

---

## Current State

The project already has:

* A Torch OSQP implementation path in `pygranso/private/torchOSQP.py`.
* A dense LU boundary in `pygranso/private/torchLinearSolve.py`.
* Workspace ownership in `pygranso/private/osqpWorkspace.py`.
* Pipeline requirements in `docs/FULL_DEVELOPMENT_AND_VALIDATION_PIPELINE.md`.

The missing part is:

* A clean direct-kernel boundary that separates validation, KKT assembly, ADMM iteration, and result construction.
* Explicit input/output structures for ADMM diagnostics.
* Tests proving that repeated ADMM iterations do not refactorize when only vector state changes.

---

## New Components to Add

### Component 1

```text
KKTSystemBuilder
```

Responsibility:

```text
Validate dense QP tensors and assemble the quasi-definite KKT matrix used by the Torch-direct path.
```

### Component 2

```text
DirectADMMRunner
```

Responsibility:

```text
Run ADMM vector updates, projection, dual update, termination checks, and diagnostics using a prepared workspace.
```

### Component 3

```text
DirectSolveResultBuilder
```

Responsibility:

```text
Convert Torch iteration state into the adapter's public result object and OSQP-style info fields.
```

---

## Class / Registry Diagrams

### Diagram 1: KKT System Builder

```text
+-------------------------------------------------------------------------------+
|                                KKTSystemBuilder                                |
+-------------------------------------------------------------------------------+
|  - sigma: float                                                                |
|  - rho: Tensor                                                                 |
|  - dtype: torch.dtype                                                          |
|  - device: torch.device                                                        |
+-------------------------------------------------------------------------------+
|  + validate(P, A, q, l, u): None         --> rejects incompatible QP inputs    |
|  + build_matrix(P, A): Tensor           --> dense KKT matrix                  |
|  + build_rhs(state): Tensor             --> normalized column RHS             |
|  + signature(P, A): StructuralSignature --> workspace compatibility key       |
+-------------------------------------------------------------------------------+
```

### Diagram 2: Direct ADMM Runner

```text
+-------------------------------------------------------------------------------+
|                                DirectADMMRunner                                |
+-------------------------------------------------------------------------------+
|  - workspace: TorchOSQPWorkspace                                               |
|  - settings: TorchOSQPSettings                                                 |
|  - diagnosticsEnabled: bool                                                    |
+-------------------------------------------------------------------------------+
|  + initialize(input): ADMMState          --> creates or restores x, z, y       |
|  + iterate_once(state): ADMMState        --> linear solve + projection update  |
|  + check_termination(state): Status      --> residual/objective based status   |
|  + solve(input): DirectSolveOutput       --> complete dense Torch solve        |
+-------------------------------------------------------------------------------+
```

### Diagram 3: Result Builder

```text
+-------------------------------------------------------------------------------+
|                             DirectSolveResultBuilder                           |
+-------------------------------------------------------------------------------+
|  - statusMap: dict                                                             |
|  - backendName: string                                                         |
+-------------------------------------------------------------------------------+
|  + objective(P, q, x): Tensor            --> primal objective                  |
|  + residuals(P, A, q, l, u, x, z, y)     --> primal/dual residuals             |
|  + info(output): dict                    --> OSQP-style telemetry              |
|  + result(output): dict                  --> adapter return payload            |
+-------------------------------------------------------------------------------+
```

---

## Class Diagram Rules

* Keep the direct ADMM runner private to the Torch implementation path.
* Do not expose a public `linear_solver` option.
* Keep matrix factorization delegated to `DenseLUSolver`; the ADMM runner should not call `torch.linalg.lu_factor_ex` directly.
* Keep infeasibility and nonconvex certificates out of this phase.

---

## Data Model

```text
DirectADMMInput
  P: Tensor[n,n]
  q: Tensor[n,1]
  A: Tensor[m,n]
  l: Tensor[m,1]
  u: Tensor[m,1]
  settings: TorchOSQPSettings
  workspace: TorchOSQPWorkspace
  validation_mode: bool
```

```text
ADMMState
  x: Tensor[n,1]
  z: Tensor[m,1]
  y: Tensor[m,1]
  x_tilde: Tensor[n,1]
  z_tilde: Tensor[m,1]
  iteration: int
  rho: Tensor or scalar
```

```text
DirectSolveOutput
  status: string
  x: Tensor[n,1]
  y: Tensor[m,1]
  z: Tensor[m,1]
  objective: float
  pri_res: float
  dua_res: float
  iter: int
  info: dict
```

---

## Storage / State

* Store persistent warm state only in `TorchOSQPWorkspace`.
* Store per-iteration temporaries in local variables or an `ADMMState` holder.
* Reuse the LU factorization when `P`, `A`, `rho`, `sigma`, dtype, device, backend, and structural signature remain compatible.
* Refactorize when the KKT matrix changes.
* Never store ADMM state in module globals.

---

## Required Methods

* `_validate_torch_qp_inputs(P, q, A, l, u, settings)`.
* `_build_kkt_matrix(P, A, sigma, rho)`.
* `_build_kkt_rhs(P, q, A, state, settings)`.
* `_project_box(v, l, u)`.
* `_admm_iteration(state, factors, input)`.
* `_compute_residuals(P, q, A, l, u, state)`.
* `_torch_direct_solve(input)`.
* `_build_torch_result(output)`.

---

## Validation Rules

* Reject NaNs and nonfinite finite-bound values.
* Permit infinite bounds for OSQP-style box constraints.
* Normalize vector RHS values to column tensors.
* Reject incompatible shapes, devices, and dtypes.
* Reject `l > u`.
* Reject material asymmetry in `P`; symmetrize only within dtype-aware tolerance.
* Check LU factorization status and solution finiteness through `DenseLUSolver`.
* Compute expensive residual diagnostics only in validation/debug mode unless needed for termination.
* Report numerical failures as solver failures, never as infeasibility.

---

## UI / API Integration

* Keep public selection through `osqp_algebra={"auto","builtin","torch"}`.
* Do not expose the direct kernel or LU object as public API.
* Return complete backend telemetry through existing QP result/info dictionaries.
* Preserve PyGRANSO outer fallback behavior for explicit `torch` failures.

---

## Workflow

1. Convert user inputs to dense Torch tensors with a shared validation path.
2. Restore compatible warm `x`, `z`, and `y` from the workspace.
3. Build or reuse the KKT factorization through `DenseLUSolver`.
4. For each ADMM iteration:
   1. Build the vector RHS.
   2. Solve with cached LU factors.
   3. Apply relaxation and box projection.
   4. Update the scaled/unscaled dual variables using the preserved equations.
   5. Check termination at the configured interval.
5. Persist compatible final `x`, `z`, and `y` back to the workspace.
6. Return OSQP-style status, objective, residual, and diagnostic telemetry.

---

## Files to Create

* `tests/test_torch_osqp_direct_admm.py`: direct-kernel tests for KKT assembly, updates, termination, and finite behavior.

---

## Files to Modify

* `pygranso/private/torchOSQP.py`: split the direct solve into validation, KKT, iteration, and result helpers.
* `pygranso/private/osqpWorkspace.py`: expose any missing workspace hooks required by the direct kernel.
* `pygranso/private/torchLinearSolve.py`: tighten RHS normalization or diagnostics if needed by ADMM tests.
* `docs/TORCH_OSQP_COMPLETION_AUDIT.md`: record implementation status after code work.

---

## Error Handling

* Raise `ValueError` for invalid user inputs.
* Raise `TorchLinearSolveError` for factorization and solve failures.
* Raise a Torch OSQP numerical error for nonfinite iterates, residuals, or objective values.
* For `auto`, let the adapter convert Torch failure into warned builtin fallback telemetry.
* For explicit `torch`, surface the failure without silently switching backend.

---

## Testing Checklist

- [x] KKT matrix block dimensions match `(n+m, n+m)`.
- [x] RHS vector inputs are normalized to `(n+m, 1)`.
- [x] ADMM projection handles finite and infinite bounds.
- [x] Dual update matches the preserved OSQP equations.
- [x] Residual and objective diagnostics are finite on supported feasible convex QPs.
- [x] LU factorization count does not increase for vector-only ADMM iterations.
- [x] Explicit `torch` numerical failure raises.
- [x] `auto` numerical failure produces causal fallback telemetry.

---

## Acceptance Criteria

* Dense Torch-direct ADMM solves supported feasible convex QPs with float64 as the authoritative path.
* The implementation reuses LU factors across ADMM iterations.
* Numerical failures are never labeled infeasible.
* Tests cover input validation, KKT assembly, projection, dual update, residuals, and telemetry.
