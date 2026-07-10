# Phase 1.3 Implementation Plan: Settings Validation and Migration Errors

## Goal

Implement **settings normalization, numerical validation, and legacy migration errors**.

This feature should allow the system to:

1. Apply common builtin/Torch OSQP defaults.
2. Reject invalid numerical inputs before backend execution.
3. Reject archived CG/CUDA Graph settings with clear migration guidance.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

---

## Current State

The project already has:

* `DEFAULT_OSQP_SETTINGS`: common adapter defaults.
* `_normalize_options(...)`: merges user options with defaults.
* `_validate_settings(...)`: checks settings ranges.
* `_validate_qp(...)`: validates Torch QP tensors.
* `LEGACY_TORCH_SETTINGS`: archived option names.

The missing part is:

* A single plan describing defaults, validation, and migration behavior.
* Clear type shapes for settings and diagnostics.
* A complete checklist of numerical validation rules.

---

## New Components to Add

### Component 1

```text
OSQPSettingsContract
```

Responsibility:

```text
Normalize and validate shared settings used by both builtin and Torch routes.
```

### Component 2

```text
NumericalInputValidator
```

Responsibility:

```text
Validate finite matrices, bounds, dtype/device compatibility, symmetry, and optional convexity diagnostics.
```

---

## Class / Registry Diagrams

### Diagram 1: OSQP Settings Contract

```text
+-------------------------------------------------------------------------------+
|                               OSQPSettingsContract                             |
+-------------------------------------------------------------------------------+
|  - rho: float                                                                  |
|  - sigma: float                                                                |
|  - alpha: float                                                                |
|  - maxIter: number                                                             |
|  - epsAbs: float                                                               |
|  - epsRel: float                                                               |
|  - scaling: number                                                             |
|  - adaptiveRho: boolean                                                        |
|  - polishing: boolean                                                          |
|  - warmStart: boolean                                                          |
+-------------------------------------------------------------------------------+
|  + defaults(dtype): dict                  --> Returns dtype-aware defaults     |
|  + merge(userSettings): dict              --> Applies user overrides safely    |
|  + validate(settings): void               --> Checks ranges and types          |
+-------------------------------------------------------------------------------+
```

### Diagram 2: Migration Error Report

```text
+-------------------------------------------------------------------------------+
|                               MigrationErrorReport                             |
+-------------------------------------------------------------------------------+
|  - legacyKeys: string[]                                                        |
|  - message: string                                                             |
|  - archiveBranch: string                                                       |
|  - replacementPath: string                                                     |
+-------------------------------------------------------------------------------+
|  + fromSettings(settings): MigrationErrorReport | null --> Detects old keys    |
|  + raise(): void                         --> Raises actionable migration error |
+-------------------------------------------------------------------------------+
```

### Diagram 3: Numerical Input Validator

```text
+-------------------------------------------------------------------------------+
|                              NumericalInputValidator                           |
+-------------------------------------------------------------------------------+
|  - dtype: torch.dtype                                                          |
|  - device: torch.device                                                        |
|  - symmetryToleranceMultiplier: float                                          |
|  - checkConvexity: boolean                                                     |
+-------------------------------------------------------------------------------+
|  + validateFinite(P,q,A): void            --> Rejects NaN/Inf matrices         |
|  + validateBounds(l,u): void              --> Rejects NaN and l > u            |
|  + validateSymmetry(P): Tensor            --> Symmetrizes only within tolerance|
|  + diagnoseConvexity(P): void             --> Optional eigvalsh PSD check      |
+-------------------------------------------------------------------------------+
```

---

## Class Diagram Rules

* Settings validation should normalize common defaults before backend-specific solve code runs.
* Legacy research options should fail with actionable migration guidance after warning policy is complete.
* Tolerance selection should depend on dtype and documented support level, not on backend preference.

---

## Data Model

```ts
type OSQPSettings = {
  id: string;
  name: string;
  rho: number;
  sigma: number;
  alpha: number;
  max_iter: number;
  eps_abs: number;
  eps_rel: number;
  check_termination: number;
  scaling: number;
  adaptive_rho: boolean;
  polishing: boolean;
  warm_start: boolean;
};
```

---

## Storage / State

This feature is stateless. It normalizes one settings object per solve and does not persist data.

Temporary state:

* Normalized settings dictionary.
* Optional symmetry/convexity diagnostics.

---

## Required Methods

```ts
function normalizeSettings(options: dict, dtype: torch.dtype): OSQPSettings
```

```ts
function validateNumericalInput(problem, settings): ValidatedProblem
```

---

## Validation Rules

Before saving or processing data, check:

1. `rho`, `sigma`, tolerances, and iteration counts are positive where required.
2. `alpha` is within the supported relaxation range.
3. Dtype is float32 or float64.
4. Devices are compatible.
5. `P`, `q`, and `A` are finite.
6. `l` and `u` contain no NaN and satisfy `l <= u`.
7. Material asymmetry in `P` is rejected.
8. Legacy settings raise migration errors.

---

## UI / API Integration

This feature is internal.

Callers:

* Adapter option normalization.
* Torch direct solver validation.
* Builtin route symmetry validation.
* Unit tests.

---

## Workflow

1. Receive user options.
2. Detect legacy settings.
3. Merge defaults by dtype.
4. Validate setting ranges.
5. Validate QP tensors and bounds.
6. Symmetrize only when safe.
7. Optionally run convexity diagnostic.
8. Return validated settings/problem or raise a clear error.

---

## Files to Create

```text
None
```

---

## Files to Modify

```text
pygranso/private/osqpTorchAdapter.py
pygranso/private/torchOSQP.py
tests/test_torch_osqp_direct.py
tests/test_torch_osqp_policy.py
```

---

## Error Handling

Handle these cases:

* Unknown setting type.
* Legacy setting key.
* Nonfinite matrix data.
* Invalid bounds.
* Unsupported dtype.
* Device mismatch.
* Material asymmetry.
* Negative eigenvalue when convexity diagnostic is enabled.

---

## Testing Checklist

Test the following:

* [ ] Float64 defaults use `1e-8`.
* [ ] Float32 defaults use `1e-5`.
* [ ] Legacy setting raises.
* [ ] NaN in matrix raises.
* [ ] NaN in bounds raises.
* [ ] `l > u` raises.
* [ ] Near-symmetric `P` is symmetrized.
* [ ] Materially asymmetric `P` is rejected.
* [ ] Optional convexity diagnostic rejects indefinite `P`.

---

## Acceptance Criteria

This phase is complete when:

1. `OSQPSettingsContract` behavior is implemented through adapter helpers.
2. Invalid numerical input cannot reach LU or builtin OSQP setup.
3. Legacy settings produce actionable errors.
4. Tests confirm defaults and validation rules.
