# Phase 4.3 Implementation Plan: Platform Gates and Backend Promotion

## Goal

Implement **backend-by-backend platform support gates and promotion policy**.

This feature should allow the system to:

1. Promote CPU, CUDA, ROCm, and MPS support independently.
2. Require real hardware evidence before claiming accelerator support.
3. Enforce conservative size, memory, accuracy, and performance gates before `auto` selects an accelerator backend.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

---

## Current State

The project already has:

* Backend policy requirements for CPU, CUDA, ROCm, and MPS.
* A support matrix requirement in the full pipeline document.
* Planned stability evidence outputs from Phase 4.2.

The missing part is:

* A concrete backend promotion record and support matrix workflow.
* Real-hardware gates for claimed accelerator support.
* Runtime policy checks that match the documented promotion state.

---

## New Components to Add

### Component 1

```text
SupportMatrixEntry
```

Responsibility:

```text
Represent one backend/dtype/device support claim and the evidence required for promotion.
```

### Component 2

```text
BackendPromotionRecord
```

Responsibility:

```text
Capture test, stability, hardware, and performance results that justify one backend promotion decision.
```

### Component 3

```text
HardwareRunnerGate
```

Responsibility:

```text
Ensure claimed accelerator support is validated on real hardware, not inferred from CPU-only tests.
```

### Component 4

```text
AutoPromotionPerformanceGate
```

Responsibility:

```text
Require representative end-to-end median runtime no worse than 5x builtin CPU OSQP before auto-promoting an accelerator.
```

---

## Class / Registry Diagrams

### Diagram 1: Support Matrix Entry

```text
+-------------------------------------------------------------------------------+
|                               SupportMatrixEntry                               |
+-------------------------------------------------------------------------------+
|  - backend: string                                                             |
|  - device: string                                                              |
|  - dtype: string                                                               |
|  - claimed: bool                                                               |
|  - autoEligible: bool                                                          |
|  - maxKktDim: int                                                              |
|  - accuracyConditioningLimit: float                                            |
|  - evidencePath: string                                                        |
+-------------------------------------------------------------------------------+
|  + can_select_auto(request): boolean    --> runtime auto-selection eligibility |
|  + reason_if_blocked(request): string   --> fallback/skip explanation          |
+-------------------------------------------------------------------------------+
```

### Diagram 2: Promotion Record

```text
+-------------------------------------------------------------------------------+
|                              BackendPromotionRecord                            |
+-------------------------------------------------------------------------------+
|  - supportEntry: SupportMatrixEntry                                            |
|  - stabilitySummaryPath: string                                                |
|  - hardwareDescription: dict                                                   |
|  - performanceRatio: float                                                     |
|  - unexplainedFailures: int                                                    |
+-------------------------------------------------------------------------------+
|  + eligible_for_claim(): boolean        --> documentation support claim        |
|  + eligible_for_auto(): boolean         --> runtime auto-promotion decision    |
|  + render_decision_log(): string        --> human-readable promotion note      |
+-------------------------------------------------------------------------------+
```

### Diagram 3: Hardware Runner Gate

```text
+-------------------------------------------------------------------------------+
|                                HardwareRunnerGate                              |
+-------------------------------------------------------------------------------+
|  - backend: string                                                             |
|  - requiredHardware: string                                                    |
|  - workflowName: string                                                        |
+-------------------------------------------------------------------------------+
|  + detect(): HardwareInfo                --> records actual runner hardware    |
|  + run_required_tests(): GateReport      --> backend-specific validation       |
|  + skip_reason(): string                 --> explicit unclaimed-backend reason |
+-------------------------------------------------------------------------------+
```

### Diagram 4: Performance Gate

```text
+-------------------------------------------------------------------------------+
|                            AutoPromotionPerformanceGate                        |
+-------------------------------------------------------------------------------+
|  - baselineBackend: string = builtin_cpu                                      |
|  - maxMedianRatio: float = 5.0                                                 |
+-------------------------------------------------------------------------------+
|  + benchmark(caseSet): PerformanceReport --> includes transfers                |
|  + passes(report): boolean              --> median runtime <= 5x baseline     |
|  + memory_preflight(request): boolean    --> conservative dense memory check   |
+-------------------------------------------------------------------------------+
```

---

## Class Diagram Rules

* Documentation support claims and runtime `auto` eligibility are related but distinct decisions.
* Promotion must be independent per backend and dtype.
* Accelerator claims require real hardware evidence.
* MPS float64 is not an auto target; it falls back to builtin CPU OSQP.

---

## Data Model

```text
SupportMatrix
  entries: list[SupportMatrixEntry]
  generated_from: list[BackendPromotionRecord]
  last_updated: date
```

```text
PerformanceReport
  backend: string
  baseline_backend: string
  case_set: string
  median_runtime_seconds: float
  baseline_median_runtime_seconds: float
  median_ratio: float
  includes_transfers: bool
  memory_preflight_passed: bool
```

---

## Storage / State

* Store support matrix in documentation and runtime constants only after evidence review.
* Store promotion records with local evidence artifacts or release documentation.
* Keep unclaimed backends explicitly documented as unclaimed.
* Do not promote based on simulated or unavailable hardware.

---

## Required Methods

* `_accelerator_capability(device, dtype, support_matrix)`.
* `_memory_preflight(n, m, dtype, device)`.
* `_auto_backend_eligible(request, support_matrix)`.
* `collect_hardware_info()`.
* `run_backend_promotion_gate(backend, dtype, device)`.
* `render_support_matrix(records)`.

---

## Validation Rules

* CPU Torch tests must run on Linux, Windows, and macOS before CPU support is complete.
* CUDA is initially eligible for promotion after real-runner evidence.
* ROCm remains unclaimed until a real runner is obtained.
* MPS float32 remains unclaimed until reusable LU passes on real Apple hardware.
* MPS float64 `auto` requests fall back to builtin CPU OSQP.
* `auto` accelerator selection requires `n+m <= 2400`, memory preflight pass, support claim, and median runtime no worse than 5x builtin CPU OSQP.

---

## UI / API Integration

* Runtime backend policy reads the support matrix to decide `auto` eligibility.
* Documentation support matrix mirrors the runtime support matrix.
* Fallback telemetry includes unsupported/unclaimed reason codes.
* Performance reports include transfer time.

---

## Workflow

1. Run deterministic CPU Torch tests on Linux, Windows, and macOS.
2. Run real-hardware gates for each accelerator candidate.
3. Run stability evidence buckets for backend/dtype/device combinations.
4. Run representative end-to-end performance comparisons including transfers.
5. Produce promotion records.
6. Promote only backend entries with complete evidence.
7. Update runtime support matrix and documentation.
8. Keep unclaimed backends unselected by `auto`.

---

## Files to Create

* `tests/test_torch_osqp_support_matrix.py`: support matrix and auto-eligibility tests.
* `scripts/torch_osqp_backend_promotion.py`: optional local promotion summary helper.

---

## Files to Modify

* `pygranso/private/solveQP.py`: align auto-selection policy with support matrix.
* `pygranso/private/torchOSQP.py`: expose device/dtype diagnostics required by promotion checks if needed.
* `.github/workflows/`: add or update backend gate workflows when CI is configured.
* `docs/FULL_DEVELOPMENT_AND_VALIDATION_PIPELINE.md`: update support matrix and promotion policy.
* `docs/TORCH_OSQP_COMPLETION_AUDIT.md`: record promotion decisions.

---

## Error Handling

* Unsupported or unclaimed hardware falls back under `auto` with a clear reason.
* Explicit `torch` on unsupported hardware raises or fails visibly according to adapter policy.
* Missing hardware evidence blocks support claims.
* Failed performance gate blocks `auto` promotion even when correctness tests pass.

---

## Testing Checklist

- [x] Support matrix rejects unclaimed ROCm by default.
- [x] MPS float64 `auto` falls back to builtin CPU OSQP.
- [x] Oversized KKT dimensions block `auto` accelerator selection.
- [x] Memory preflight failure blocks `auto` accelerator selection.
- [x] Explicit `torch` does not silently fallback.
- [x] Performance gate uses runtime including transfers.
- [x] Documentation and runtime support matrix stay synchronized.

---

## Acceptance Criteria

* Backend support is promoted independently and only with evidence.
* Runtime `auto` selection follows the support matrix.
* Accelerator auto-promotion requires correctness, real hardware, size/memory checks, and performance evidence.
