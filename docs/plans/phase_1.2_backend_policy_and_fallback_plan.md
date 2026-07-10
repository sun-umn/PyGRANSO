# Phase 1.2 Implementation Plan: Backend Policy and Fallback Telemetry

## Goal

Implement **backend policy and fallback telemetry**.

This feature should allow the system to:

1. Select `builtin` or `torch` from the public `osqp_algebra` contract.
2. Make every automatic fallback visible and causal.
3. Prevent explicit `torch` requests from silently changing backend.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

---

## Current State

The project already has:

* `_select_backend(...)`: chooses builtin or Torch based on request, device, size, memory, and promotion.
* `_accelerator_capability(...)`: checks CUDA, ROCm, and MPS promotion status.
* `info["fallback"]`: records automatic fallback details.

The missing part is:

* A template-based implementation plan for the policy data structures.
* Clear acceptance criteria for explicit vs automatic backend behavior.
* A testing checklist for fallback telemetry fields.

---

## New Components to Add

### Component 1

```text
BackendSelection
```

Responsibility:

```text
Represent the requested backend, selected backend, dense KKT estimate, memory estimate, and selection reason.
```

### Component 2

```text
FallbackTelemetry
```

Responsibility:

```text
Represent why an automatic request fell back, what backend was tried, and what result or exception caused the retry.
```

---

## Class / Registry Diagrams

### Diagram 1: Backend Selection

```text
+-------------------------------------------------------------------------------+
|                                BackendSelection                                |
+-------------------------------------------------------------------------------+
|  - requestedBackend: string                                                    |
|  - selectedBackend: string                                                     |
|  - selectionReason: string                                                     |
|  - estimatedKktDim: number                                                     |
|  - estimatedDenseWorkingMemoryMb: number                                       |
|  - fallback: FallbackTelemetry                                                 |
+-------------------------------------------------------------------------------+
|  + isBuiltin(): boolean                   --> True when builtin will run       |
|  + isTorch(): boolean                     --> True when Torch will run         |
|  + toInfoFields(): dict                   --> Converts to result telemetry     |
+-------------------------------------------------------------------------------+
```

### Diagram 2: Fallback Telemetry

```text
+-------------------------------------------------------------------------------+
|                                FallbackTelemetry                               |
+-------------------------------------------------------------------------------+
|  - occurred: boolean                                                           |
|  - trigger: string                                                             |
|  - requestedBackend: string                                                    |
|  - selectedBackend: string                                                     |
|  - fallbackBackend: string                                                     |
|  - deviceTransfer: boolean                                                     |
|  - status: string | null                                                       |
|  - exceptionType: string | null                                                |
|  - message: string | null                                                      |
+-------------------------------------------------------------------------------+
|  + fromSelectionPolicy(reason): FallbackTelemetry --> Builds policy fallback   |
|  + fromException(exc): FallbackTelemetry          --> Builds exception record  |
|  + fromUnsolvedStatus(info): FallbackTelemetry    --> Builds status record     |
+-------------------------------------------------------------------------------+
```

### Diagram 3: Accelerator Capability

```text
+-------------------------------------------------------------------------------+
|                              AcceleratorCapability                             |
+-------------------------------------------------------------------------------+
|  - deviceType: string                                                          |
|  - dtype: torch.dtype                                                          |
|  - promoted: boolean                                                           |
|  - supported: boolean                                                          |
|  - reason: string                                                              |
+-------------------------------------------------------------------------------+
|  + requiresFallback(): boolean            --> True if auto must use builtin    |
|  + explain(): string                      --> Returns warning reason           |
+-------------------------------------------------------------------------------+
```

---

## Class Diagram Rules

* Backend selection is an adapter policy boundary, not a solver math component.
* Fallback telemetry must explain automatic backend changes without mutating explicit requests.
* Keep support-matrix checks separate from numerical solve diagnostics.

---

## Data Model

```ts
type BackendSelectionInfo = {
  id: string;
  name: string;
  requestedBackend: "auto" | "builtin" | "torch";
  selectedBackend: "builtin" | "torch";
  selectionReason: string;
  estimatedKktDim: number;
  estimatedDenseWorkingMemoryMb: number;
  fallback?: {
    occurred: boolean;
    trigger?: string;
    message?: string;
  };
};
```

---

## Storage / State

This feature is stateless. It receives the QP request and settings, returns
selection telemetry, and does not persist data.

Temporary state:

* Backend choice for one QP solve.
* Fallback details for one QP solve.

---

## Required Methods

```ts
function selectBackend(request): BackendSelectionInfo
```

```ts
function buildFallbackTelemetry(trigger, cause): FallbackTelemetry
```

---

## Validation Rules

Before selecting a backend, check:

1. Public algebra is one of `auto`, `builtin`, or `torch`.
2. CPU `auto` uses builtin.
3. Unsupported accelerator `auto` falls back with warning.
4. Oversized automatic Torch request falls back with warning.
5. Explicit `torch` warns and attempts when oversized.
6. Explicit `torch` failure propagates.

---

## UI / API Integration

This feature is internal.

Callers:

* `solve_osqp_torch_qp(...)` calls `_select_backend(...)`.
* Tests inspect returned `info` fields.
* PyGRANSO sees failures through existing outer fallback paths.

---

## Workflow

1. Receive `osqp_algebra`, device, dtype, and QP size.
2. Estimate dense KKT dimension and memory.
3. Apply explicit request rules.
4. Apply automatic CPU/accelerator rules.
5. Run selected backend.
6. If automatic Torch fails or is unsolved, retry builtin and attach telemetry.

---

## Files to Create

```text
None
```

---

## Files to Modify

```text
pygranso/private/osqpTorchAdapter.py
tests/test_torch_osqp_policy.py
docs/FULL_DEVELOPMENT_AND_VALIDATION_PIPELINE.md
```

---

## Error Handling

Handle these cases:

* Unknown `osqp_algebra`.
* Unsupported accelerator.
* KKT dimension above automatic limit.
* Memory preflight failure.
* Torch exception under `auto`.
* Torch unsolved status under `auto`.
* Explicit Torch unsolved status.

---

## Testing Checklist

Test the following:

* [ ] CPU `auto` selects builtin.
* [ ] Explicit builtin selects builtin.
* [ ] Explicit torch selects Torch.
* [ ] Explicit torch above limit warns and attempts.
* [ ] Unsupported accelerator auto falls back.
* [ ] Torch exception auto fallback records exception type.
* [ ] Torch unsolved auto fallback records status.
* [ ] Explicit Torch failure does not fallback silently.

---

## Acceptance Criteria

This phase is complete when:

1. `BackendSelection` fields are returned in `info`.
2. `FallbackTelemetry` fully explains automatic fallback.
3. Explicit backend requests remain explicit.
4. Validation tests confirm every selection branch.
