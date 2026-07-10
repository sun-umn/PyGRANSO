# Phase 2.1 Implementation Plan: Dense LU Workspace Lifecycle

## Goal

Implement **optimizer-owned Torch OSQP workspace and reusable dense LU lifecycle**.

This feature should allow the system to:

1. Reuse LU factors across compatible ADMM iterations and vector-only solves.
2. Preserve warm `x`, `z`, `y` state only when structure is compatible.
3. Reset state safely on structure, dtype, device, order, or backend changes.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

---

## Current State

The project already has:

* `pygranso/private/osqpWorkspace.py`: defines `TorchOSQPWorkspace`.
* `pygranso/private/torchLinearSolve.py`: defines `DenseLUSolver`, `LinearSolveDiagnostics`, and `TorchLinearSolveError`.
* `pygranso/private/bfgssqp.py`: creates one workspace per BFGS-SQP run.
* `pygranso/private/torchOSQP.py`: prepares workspace and uses the LU solver.

The missing part is:

* A template-format implementation plan for the workspace/LU data model.
* Explicit ASCII class diagrams for every class/data holder.
* A detailed checkbox workflow for reuse, refactorization, and invalidation.

---

## New Components to Add

### Component 1

```text
TorchOSQPWorkspace
```

Responsibility:

```text
Own all per-optimizer Torch OSQP state, including warm vectors, signatures, scaling, rho, LU factors, builtin cache, and diagnostics.
```

### Component 2

```text
DenseLUSolver
```

Responsibility:

```text
Validate dense KKT matrices, cache PyTorch LU factors, solve repeated RHS values, and emit diagnostics.
```

### Component 3

```text
LinearSolveDiagnostics
```

Responsibility:

```text
Report factorization status, factorization/solve counts, and optional linear residuals.
```

---

## Class / Registry Diagrams

### Diagram 1: Runtime State Registry

```text
+-------------------------------------------------------------------------------+
|                               TorchOSQPWorkspace                               |
+-------------------------------------------------------------------------------+
|  - state: dict | None                                                          |
|  - problem_signature: tuple | None                                             |
|  - constraint_order_signature: tuple | None                                    |
|  - p_pattern: torch.Tensor | None                                              |
|  - a_pattern: torch.Tensor | None                                              |
|  - scaling: dict | None                                                        |
|  - rho_bar: float | None                                                       |
|  - active_backend: string | None                                               |
|  - linear_solver: DenseLUSolver                                                |
|  - builtin_cache: dict | None                                                  |
|  - builtin_stats: dict                                                         |
+-------------------------------------------------------------------------------+
|  + reset_torch(): void                    --> Clears Torch warm/factor state   |
|  + reset_builtin(): void                  --> Clears builtin cache/statistics  |
|  + reset(): void                          --> Clears all backend state         |
|  + ensure_backend(backend): boolean       --> Invalidates on backend change    |
+-------------------------------------------------------------------------------+
```

### Diagram 2: Service Class

```text
+-------------------------------------------------------------------------------+
|                                  DenseLUSolver                                 |
+-------------------------------------------------------------------------------+
|  - _matrix: torch.Tensor | None                                                |
|  - _lu: torch.Tensor | None                                                    |
|  - _pivots: torch.Tensor | None                                                |
|  - _info: number                                                               |
|  - factorization_count: number                                                 |
|  - solve_count: number                                                         |
+-------------------------------------------------------------------------------+
|  + clear(): void                         --> Drops cached matrix and factors   |
|  + is_factorized_for(matrix): boolean    --> Checks exact factor reuse         |
|  + factorize(matrix): void               --> Validates and factors K           |
|  + refactorize(matrix): void             --> Forces a new factorization        |
|  + factorize_if_needed(matrix): boolean  --> Reuses or factors as needed       |
|  + solve(rhs, calculate_residual): tuple --> Solves RHS and returns diagnostics|
+-------------------------------------------------------------------------------+
```

### Diagram 3: Data Transfer Object

```text
+-------------------------------------------------------------------------------+
|                              LinearSolveDiagnostics                            |
+-------------------------------------------------------------------------------+
|  - solver: string                                                              |
|  - factorization_info: number                                                  |
|  - factorization_count: number                                                 |
|  - solve_count: number                                                         |
|  - linear_residual_norm: float | None                                          |
|  - relative_linear_residual: float | None                                      |
+-------------------------------------------------------------------------------+
|  + as_dict(): dict                        --> Converts diagnostics to info     |
+-------------------------------------------------------------------------------+
```

### Diagram 4: Error Class

```text
+-------------------------------------------------------------------------------+
|                              TorchLinearSolveError                             |
+-------------------------------------------------------------------------------+
|  - message: string                                                             |
+-------------------------------------------------------------------------------+
|  + __init__(message): void                --> Wraps LU factor/solve failures   |
+-------------------------------------------------------------------------------+
```

---

## Class Diagram Rules

* `TorchOSQPWorkspace` owns mutable solver state for exactly one BFGS-SQP run.
* `DenseLUSolver` owns factorization lifecycle and should be the only object calling PyTorch LU primitives.
* Diagnostics are internal telemetry and should not become module-global state.

---

## Data Model

```ts
type WorkspaceState = {
  id: string;
  name: string;
  x?: "Tensor[n]";
  z?: "Tensor[m]";
  y?: "Tensor[m]";
  problemSignature?: unknown[];
  constraintOrderSignature?: unknown[];
  activeBackend?: "builtin" | "torch";
  rhoBar?: number;
  createdAt?: string;
  updatedAt?: string;
};
```

---

## Storage / State

### Temporary State

Use in-memory state owned by one `AlgBFGSSQP` run.

Use this for:

* Warm vectors `x`, `z`, `y`
* Problem signatures
* Scaling cache
* Adaptive rho state
* LU factors
* Builtin OSQP cache
* Last diagnostics

No workspace state should be module-global or shared across independent runs.

---

## Required Methods

```ts
function prepareWorkspace(workspace, P, A, orderSignature): void
```

```ts
function factorizeIfNeeded(K): boolean
```

```ts
function solve(rhs): [Tensor, LinearSolveDiagnostics]
```

---

## Validation Rules

Before reusing state, check:

1. Matrix shapes match.
2. Constraint order signature matches.
3. Structural patterns match.
4. Dtype and device match.
5. Backend has not changed.
6. Matrix values are unchanged if LU is reused.
7. RHS is finite and shape-compatible.

---

## UI / API Integration

This feature is purely internal.

Callers:

* `AlgBFGSSQP` creates the workspace.
* `solveQP` passes workspace through OSQP options.
* `solve_torch_osqp_direct` consumes and updates workspace state.

---

## Workflow

1. BFGS-SQP creates a workspace.
2. Adapter selects backend and calls `ensure_backend`.
3. Torch solver validates QP and prepares workspace.
4. Compatible state is reused.
5. Incompatible state is reset.
6. KKT is factorized or reused.
7. ADMM iterations solve repeated RHS values.
8. Final warm state and diagnostics are stored.

---

## Files to Create

```text
None
```

---

## Files to Modify

```text
pygranso/private/osqpWorkspace.py
pygranso/private/torchLinearSolve.py
pygranso/private/torchOSQP.py
pygranso/private/bfgssqp.py
tests/test_torch_linear_solve.py
tests/test_osqp_workspace_lifecycle.py
```

---

## Error Handling

Handle these cases:

* LU factorization is unavailable.
* LU reports singular matrix.
* LU factors contain nonfinite values.
* RHS is nonfinite.
* Solve returns nonfinite values.
* Workspace backend changes.
* User passes a non-workspace object.

---

## Testing Checklist

Test the following:

* [ ] Workspace starts empty.
* [ ] Compatible vector update reuses LU.
* [ ] Matrix-value update preserves warm state and refactorizes.
* [ ] Rho change refactorizes.
* [ ] Sigma change refactorizes.
* [ ] Structure change clears warm state.
* [ ] Backend change clears both backend caches.
* [ ] Vector RHS shape is restored.
* [ ] Nonfinite matrix/RHS is rejected.
* [ ] Singular KKT raises a clear error.

---

## Acceptance Criteria

This phase is complete when:

1. `TorchOSQPWorkspace` owns all reusable solver state.
2. `DenseLUSolver` implements safe factorize/solve/refactorize behavior.
3. `LinearSolveDiagnostics` is available in solver info.
4. Tests confirm reuse, refactorization, and invalidation behavior.
5. No global warm state remains.
