# Phase 3.2 Implementation Plan: PyGRANSO Integration

## Goal

Implement **end-to-end PyGRANSO integration for the revised Torch-OSQP path**.

This feature should allow the system to:

1. Route BFGS-SQP QP subproblems through `osqp_algebra={"auto","builtin","torch"}`.
2. Preserve PyGRANSO steering, stationarity, penalty update, and fallback semantics.
3. Validate complete constrained optimization runs, not only isolated QP solves.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

---

## Current State

The project already has:

* BFGS-SQP integration code in `pygranso/private/bfgssqp.py`.
* QP solve routing in `pygranso/private/solveQP.py`.
* Steering and termination logic in private PyGRANSO modules.
* Documentation requiring B1/B2/B3 and complete constrained optimization coverage.

The missing part is:

* A precise bridge contract between BFGS-SQP and the revised dense Torch QP solver.
* Integration tests that verify PyGRANSO-level behavior under builtin, Torch, and fallback paths.
* Explicit handling of workspace lifetime per BFGS-SQP run.

---

## New Components to Add

### Component 1

```text
SolveQPRequest
```

Responsibility:

```text
Package the QP matrices, bounds, requested backend, optimizer device, tolerances, and workspace for one PyGRANSO subproblem.
```

### Component 2

```text
AlgBFGSSQPOSQPBridge
```

Responsibility:

```text
Own the per-run workspace and pass it through PyGRANSO QP solve calls without module globals.
```

### Component 3

```text
PyGRANSOFallbackContract
```

Responsibility:

```text
Define how automatic Torch fallback and existing outer PyGRANSO fallback behavior interact.
```

---

## Class / Registry Diagrams

### Diagram 1: QP Request

```text
+-------------------------------------------------------------------------------+
|                                  SolveQPRequest                                |
+-------------------------------------------------------------------------------+
|  - P: Tensor or array                                                          |
|  - q: Tensor or array                                                          |
|  - A: Tensor or array                                                          |
|  - l: Tensor or array                                                          |
|  - u: Tensor or array                                                          |
|  - osqp_algebra: string                                                        |
|  - target_device: device                                                       |
|  - workspace: TorchOSQPWorkspace                                               |
|  - settings: CommonOSQPSettings                                                |
+-------------------------------------------------------------------------------+
|  + validate_shapes(): None              --> QP adapter shape checks            |
|  + backend_policy_input(): dict         --> auto/builtin/torch selection data  |
|  + solver_payload(): dict               --> backend-specific solve payload     |
+-------------------------------------------------------------------------------+
```

### Diagram 2: BFGS-SQP Bridge

```text
+-------------------------------------------------------------------------------+
|                              AlgBFGSSQPOSQPBridge                              |
+-------------------------------------------------------------------------------+
|  - bfgsRunId: string                                                           |
|  - torchWorkspace: TorchOSQPWorkspace                                          |
|  - options: pygransoStruct                                                     |
+-------------------------------------------------------------------------------+
|  + create_workspace(): TorchOSQPWorkspace  --> called once per run             |
|  + build_request(subproblem): SolveQPRequest --> packages QP data              |
|  + solve(request): QPResult                --> calls solveQP                   |
|  + finalize(result): None                  --> records diagnostics             |
+-------------------------------------------------------------------------------+
```

### Diagram 3: Fallback Contract

```text
+-------------------------------------------------------------------------------+
|                              PyGRANSOFallbackContract                          |
+-------------------------------------------------------------------------------+
|  - requestedBackend: string                                                    |
|  - selectedBackend: string                                                     |
|  - outerFallbackAllowed: bool                                                  |
+-------------------------------------------------------------------------------+
|  + auto_fallback_allowed(): boolean      --> true only for osqp_algebra=auto   |
|  + explicit_torch_behavior(): string     --> raise and let outer strategy act  |
|  + telemetry_fields(): dict              --> complete causal fallback payload  |
+-------------------------------------------------------------------------------+
```

---

## Class Diagram Rules

* One `TorchOSQPWorkspace` belongs to one BFGS-SQP run.
* The bridge may pass workspace references but must not make them global.
* PyGRANSO-level tests should verify optimization outcomes, not only QP-level status.
* Explicit `torch` failures are visible to PyGRANSO; automatic fallback is an adapter decision only for `auto`.

---

## Data Model

```text
QPSubproblemRecord
  iteration: int
  phase: string
  dimensions: tuple[int,int]
  backend_requested: string
  backend_selected: string
  fallback: dict
  qp_status: string
  steering_state: dict
```

```text
PyGRANSOIntegrationResult
  final_x: Tensor
  final_f: float
  constraint_violation: float
  stationarity: float
  qp_records: list[QPSubproblemRecord]
  termination_code: int
```

---

## Storage / State

* Store the workspace inside the BFGS-SQP run object or closure.
* Store per-QP records in diagnostics, not module globals.
* Reset workspace on backend, dtype, device, dimension, order, or structure changes.
* Preserve compatible warm state across QP subproblems from the same optimizer run.

---

## Required Methods

* `_create_torch_osqp_workspace_for_run(options)`.
* `_build_solve_qp_request(subproblem, options, workspace)`.
* `_solve_qp_with_backend_policy(request)`.
* `_record_qp_backend_telemetry(result, run_state)`.
* `_handle_explicit_torch_failure(error, run_state)`.
* `_verify_pygranso_stationarity(result)`.

---

## Validation Rules

* CPU target with `auto` selects builtin OSQP.
* Validated accelerator target with `auto` may select Torch.
* Unsupported accelerator, oversized KKT, failed Torch solve, or unsolved Torch result under `auto` falls back with warning and causal telemetry.
* Explicit `torch` never silently changes backend.
* MPS float64 `auto` requests fall back to builtin CPU OSQP.
* PyGRANSO steering, penalty updates, B1/B2/B3, stationarity, and termination behavior remain compatible with the existing solver path.

---

## UI / API Integration

* Preserve the public `osqp_algebra` option.
* Do not add a public `linear_solver` option.
* Keep legacy CG/CUDA Graph settings on the migration-warning/error path from Phase 1.3.
* Surface backend/fallback telemetry in existing diagnostic structures where possible.

---

## Workflow

1. BFGS-SQP creates a private Torch OSQP workspace at run initialization.
2. Each QP subproblem builds a `SolveQPRequest`.
3. The backend policy selects builtin or Torch.
4. The selected backend solves or raises.
5. `auto` Torch failure retries builtin with complete telemetry.
6. Explicit `torch` failure is returned to PyGRANSO's existing outer fallback/error strategy.
7. The optimizer records QP telemetry and proceeds through steering, penalty, and termination checks.
8. End-to-end tests validate final optimization behavior.

---

## Files to Create

* `tests/test_pygranso_torch_osqp_integration.py`: constrained optimization tests for builtin, Torch, and fallback paths.

---

## Files to Modify

* `pygranso/private/bfgssqp.py`: create and own the Torch OSQP workspace.
* `pygranso/private/solveQP.py`: route requests through the revised backend policy.
* `pygranso/private/qpSteeringStrategy.py`: add telemetry assertions only if needed.
* `pygranso/private/qpTerminationCondition.py`: add status handling only if needed.
* `docs/TORCH_OSQP_COMPLETION_AUDIT.md`: record PyGRANSO integration evidence.

---

## Error Handling

* Preserve current PyGRANSO outer fallback behavior.
* Do not swallow explicit `torch` failures.
* Warn on automatic builtin fallback and include cause, attempted backend, selected backend, dimensions, dtype, device, and exception/status.
* Treat numerical failure as numerical failure, not infeasibility.

---

## Testing Checklist

- [x] CPU `auto` path uses builtin OSQP.
- [x] Explicit `builtin` path remains unchanged.
- [x] Explicit `torch` path does not silently fall back.
- [x] Automatic Torch failure retries builtin with causal telemetry.
- [x] Workspace is created once per BFGS-SQP run.
- [x] Warm state is reused across compatible QP subproblems.
- [x] Steering and penalty update tests still pass.
- [x] B1/B2/B3 behavior remains covered.
- [x] Complete constrained optimization examples pass with supported backends.

---

## Acceptance Criteria

* PyGRANSO can use the revised Torch-OSQP path through the existing public option.
* Workspace lifetime is per optimizer run.
* Automatic fallback and explicit backend behavior are both observable and tested.
* Complete constrained optimization tests pass for the supported backend matrix.
