# Phase 4.1 Implementation Plan: Test Gates and Differential Validation

## Goal

Implement **deterministic unit, differential, metamorphic, randomized, and PyGRANSO end-to-end validation gates**.

This feature should allow the system to:

1. Prove supported dense Torch-OSQP behavior inside the documented size, dtype, conditioning, and backend matrix.
2. Compare Torch and builtin OSQP by status compatibility, feasibility, stationarity, residuals, and normalized objective gap.
3. Run deterministic core tests on every change and broader randomized/stress suites nightly and before release.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

---

## Current State

The project already has:

* Existing PyGRANSO tests for optimization behavior.
* Torch-OSQP development requirements in the full pipeline document.
* A completion audit that tracks which areas still need evidence.

The missing part is:

* A phase-specific test architecture covering solver internals, adapter behavior, and PyGRANSO integration.
* Reproducible randomized case families with fixed seeds.
* Explicit release gates for supported support claims.

---

## New Components to Add

### Component 1

```text
TorchOSQPUnitGate
```

Responsibility:

```text
Collect deterministic unit tests for validation, KKT assembly, LU reuse, ADMM updates, scaling, adaptive rho, polishing, and workspace invalidation.
```

### Component 2

```text
DifferentialGate
```

Responsibility:

```text
Run Torch and builtin OSQP with identical settings and compare backend-neutral metrics.
```

### Component 3

```text
MetamorphicCaseFamily
```

Responsibility:

```text
Generate deterministic equivalent QP transformations that should preserve solution quality and status compatibility.
```

### Component 4

```text
PyGRANSOEndToEndGate
```

Responsibility:

```text
Validate that the revised QP path preserves steering, stationarity, penalty, fallback, and full optimization behavior.
```

---

## Class / Registry Diagrams

### Diagram 1: Unit Gate Registry

```text
+-------------------------------------------------------------------------------+
|                                TorchOSQPUnitGate                               |
+-------------------------------------------------------------------------------+
|  - categories: list[string]                                                    |
|  - requiredOnEveryChange: bool                                                 |
+-------------------------------------------------------------------------------+
|  + collect(): list[TestModule]            --> unit test modules                |
|  + run_core(): TestReport                 --> deterministic fast gate          |
|  + assert_zero_unexplained_failures(): None --> supported-matrix requirement   |
+-------------------------------------------------------------------------------+
```

### Diagram 2: Differential Gate

```text
+-------------------------------------------------------------------------------+
|                                DifferentialGate                                |
+-------------------------------------------------------------------------------+
|  - commonSettings: CommonOSQPSettings                                          |
|  - tolerances: ToleranceSpec                                                   |
|  - caseFamilies: list[QPCaseFamily]                                            |
+-------------------------------------------------------------------------------+
|  + solve_both(case): PairResult          --> builtin and Torch results         |
|  + compare(pair): ComparisonReport       --> backend-neutral metrics           |
|  + explain_failure(report): string       --> reproducible failure summary      |
+-------------------------------------------------------------------------------+
```

### Diagram 3: Metamorphic Case Family

```text
+-------------------------------------------------------------------------------+
|                              MetamorphicCaseFamily                             |
+-------------------------------------------------------------------------------+
|  - baseSeed: int                                                               |
|  - transformations: list[string]                                               |
+-------------------------------------------------------------------------------+
|  + generate(seed): QPCase                --> deterministic feasible convex QP  |
|  + transform(case): list[QPCase]         --> permuted/scaled/equivalent cases  |
|  + expected_relation(a, b): Relation     --> status/objective feasibility rule |
+-------------------------------------------------------------------------------+
```

### Diagram 4: PyGRANSO End-to-End Gate

```text
+-------------------------------------------------------------------------------+
|                              PyGRANSOEndToEndGate                              |
+-------------------------------------------------------------------------------+
|  - scenarios: list[OptimizationScenario]                                       |
|  - backends: list[string]                                                      |
+-------------------------------------------------------------------------------+
|  + run_scenario(scenario, backend): Result --> full optimizer run              |
|  + check_steering(result): None          --> steering and penalty assertions   |
|  + check_stationarity(result): None      --> final stationarity assertion      |
+-------------------------------------------------------------------------------+
```

---

## Class Diagram Rules

* Gate classes are conceptual test organization boundaries, not required runtime APIs.
* Deterministic gates must run quickly enough for every change.
* Nightly randomized gates must use fixed seeds and write reproducible failure data.
* Differential tests must not compare raw iterates or iteration counts.

---

## Data Model

```text
QPCase
  case_id: string
  n: int
  m: int
  dtype: dtype
  device: device
  conditioning_estimate: float
  P: Tensor
  q: Tensor
  A: Tensor
  l: Tensor
  u: Tensor
  seed: int
```

```text
TestGateReport
  gate_name: string
  backend: string
  case_count: int
  passed: int
  failed: int
  unexplained_failed: int
  duration_seconds: float
  artifact_paths: list[string]
```

---

## Storage / State

* Store deterministic test cases in test fixtures or generated helper functions.
* Store nightly artifacts locally until a later release step publishes them.
* Serialize every randomized failure with input data, settings, seed, platform, and backend.
* Do not store generated artifacts in source-controlled package paths unless explicitly approved.

---

## Required Methods

* `make_feasible_convex_qp(seed, n, m, dtype, device, conditioning)`.
* `make_metamorphic_variants(case)`.
* `run_torch_builtin_differential(case, settings)`.
* `assert_status_feasibility_stationarity_objective(report)`.
* `serialize_failure_reproduction(case, result_pair, report)`.
* `run_pygranso_backend_scenario(scenario, backend)`.

---

## Validation Rules

* Every supported unit category must have deterministic coverage.
* Nightly family/backend buckets use 100 fixed reproducible seeds.
* Nightly suites must finish within two hours.
* Supported cases require zero unexplained failures.
* Conditioning up to `1e8` is in the supported accuracy target.
* Cases approaching `1e10` are stress evidence, not guaranteed support.
* Tests must cover Python 3.10+ with PyTorch 2.8 and the current supported stable PyTorch release.

---

## UI / API Integration

* Integrate deterministic tests into the normal test command or CI path.
* Integrate stress/randomized gates into nightly and pre-release commands.
* Expose failure artifact paths in test output.
* Keep test helpers private to the test suite.

---

## Workflow

1. Run deterministic unit tests for validation, LU, KKT, ADMM, scaling, polishing, workspace, and telemetry.
2. Run deterministic differential tests under identical builtin/Torch settings.
3. Run PyGRANSO steering and full optimization scenarios.
4. Generate nightly randomized and metamorphic cases from fixed seed lists.
5. Serialize every failure with full reproduction data.
6. Summarize gate results by backend, dtype, conditioning bucket, and device.
7. Block promotion or release on unexplained supported-case failures.

---

## Files to Create

* `tests/test_torch_osqp_validation.py`.
* `tests/test_torch_osqp_linear_solve.py`.
* `tests/test_torch_osqp_admm.py`.
* `tests/test_torch_osqp_workspace.py`.
* `tests/test_torch_osqp_differential.py`.
* `tests/test_torch_osqp_metamorphic.py`.
* `tests/test_pygranso_torch_osqp_end_to_end.py`.

---

## Files to Modify

* `tests/conftest.py`: add deterministic seed/device fixtures if needed.
* `pyproject.toml` or equivalent test configuration: add markers for deterministic, nightly, stress, and hardware gates if needed.
* `docs/TORCH_OSQP_COMPLETION_AUDIT.md`: record gate completion status.

---

## Error Handling

* Any unexplained supported-case failure blocks release.
* Unsupported hardware should skip with a clear reason, not fail silently.
* Randomized failures must include serialized reproduction data.
* Test helpers should fail fast on invalid case generation.

---

## Testing Checklist

- [x] Input validation tests cover NaNs, `l > u`, shape/device/dtype mismatch, and material asymmetry.
- [x] LU tests cover reuse, refactorization, singular matrices, nonfinite RHS, and RHS shapes.
- [x] ADMM tests cover KKT assembly, projection, dual updates, residuals, and termination.
- [x] Scaling/adaptive/polishing tests cover success and failure paths.
- [x] Workspace invalidation tests cover all documented invalidation triggers.
- [x] Differential tests compare metrics, not iterates.
- [x] Metamorphic tests use deterministic equivalent transformations.
- [x] PyGRANSO tests cover steering, stationarity, penalty updates, B1/B2/B3, fallback, and full constrained runs.

---

## Acceptance Criteria

* Deterministic core tests can run on every change.
* Nightly randomized/stress suites have fixed seeds and local failure artifacts.
* Supported cases have zero unexplained failures.
* Test reports provide enough information to reproduce every failure.
