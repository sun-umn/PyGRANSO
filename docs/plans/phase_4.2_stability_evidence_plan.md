# Phase 4.2 Implementation Plan: Stability Evidence Package

## Goal

Implement **the local stability evidence package for Torch-OSQP validation**.

This feature should allow the system to:

1. Produce case-level CSV results, a JSON manifest, a Markdown summary, and failure reproduction data.
2. Record commit, platform, hardware, Python, PyTorch, OSQP, backend, settings, and seeds.
3. Keep artifacts local unless the user explicitly approves publication.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

---

## Current State

The project already has:

* Planned deterministic and nightly gates in Phase 4.1.
* Documentation requiring stability evidence outputs.
* Local workspace authorization to keep artifacts local.

The missing part is:

* A concrete artifact schema.
* A runner that writes consistent case-level results and summaries.
* A reproduction bundle for every failure.

---

## New Components to Add

### Component 1

```text
StabilityCaseResult
```

Responsibility:

```text
Represent one backend/case outcome with status, metrics, timings, and reproduction pointers.
```

### Component 2

```text
StabilityManifest
```

Responsibility:

```text
Record environment, commit, platform, hardware, package versions, settings, backends, and seed lists.
```

### Component 3

```text
FailureReproductionBundle
```

Responsibility:

```text
Serialize all data needed to replay a failed case locally.
```

### Component 4

```text
StabilityMarkdownSummary
```

Responsibility:

```text
Summarize pass/fail counts, unsupported skips, backend claims, and promotion recommendations.
```

---

## Class / Registry Diagrams

### Diagram 1: Case Result

```text
+-------------------------------------------------------------------------------+
|                               StabilityCaseResult                              |
+-------------------------------------------------------------------------------+
|  - case_id: string                                                             |
|  - family: string                                                              |
|  - backend: string                                                             |
|  - dtype: string                                                               |
|  - device: string                                                              |
|  - kkt_dim: int                                                                |
|  - conditioning_estimate: float                                                |
|  - status: string                                                              |
|  - objective_gap: float                                                        |
|  - feasibility: float                                                          |
|  - stationarity: float                                                         |
|  - reproduction_path: string                                                   |
+-------------------------------------------------------------------------------+
|  + to_csv_row(): dict                   --> case-level CSV row                 |
|  + passed_supported_gate(): boolean      --> support-matrix gate               |
+-------------------------------------------------------------------------------+
```

### Diagram 2: Manifest

```text
+-------------------------------------------------------------------------------+
|                                StabilityManifest                               |
+-------------------------------------------------------------------------------+
|  - commit: string                                                              |
|  - platform: dict                                                              |
|  - hardware: dict                                                              |
|  - python: string                                                              |
|  - pytorch: string                                                             |
|  - osqp: string                                                                |
|  - backends: list[string]                                                      |
|  - settings: dict                                                              |
|  - seeds: dict                                                                 |
+-------------------------------------------------------------------------------+
|  + collect(): StabilityManifest         --> reads local environment            |
|  + to_json(): dict                      --> manifest serialization             |
+-------------------------------------------------------------------------------+
```

### Diagram 3: Failure Reproduction Bundle

```text
+-------------------------------------------------------------------------------+
|                            FailureReproductionBundle                           |
+-------------------------------------------------------------------------------+
|  - case: QPCase                                                                |
|  - settings: dict                                                              |
|  - torch_result: dict                                                          |
|  - builtin_result: dict                                                        |
|  - comparison: dict                                                            |
|  - exception: string                                                           |
+-------------------------------------------------------------------------------+
|  + write(directory): string             --> serialized reproduction path       |
|  + replay(): ComparisonReport           --> optional local replay helper       |
+-------------------------------------------------------------------------------+
```

### Diagram 4: Markdown Summary

```text
+-------------------------------------------------------------------------------+
|                            StabilityMarkdownSummary                            |
+-------------------------------------------------------------------------------+
|  - manifest: StabilityManifest                                                 |
|  - case_results: list[StabilityCaseResult]                                     |
+-------------------------------------------------------------------------------+
|  + aggregate(): dict                    --> counts by backend/family/bucket    |
|  + recommendations(): list[string]      --> support promotion notes            |
|  + render(): string                     --> Markdown summary                   |
+-------------------------------------------------------------------------------+
```

---

## Class Diagram Rules

* Artifact writers should be deterministic for fixed inputs and seed order.
* Evidence generation should not change solver behavior.
* Failure bundles must avoid secrets and must not include unrelated local paths.
* Stability artifacts are local outputs, not package source files.

---

## Data Model

```text
torch_osqp_stability_results.csv
  case_id
  family
  seed
  backend
  dtype
  device
  n
  m
  kkt_dim
  conditioning_estimate
  status
  feasibility
  stationarity
  objective_gap
  runtime_seconds
  passed
  failure_reason
  reproduction_path
```

```text
stability_manifest.json
  commit
  platform
  hardware
  python
  pytorch
  osqp
  numpy
  backend
  settings
  seeds
  command
```

---

## Storage / State

* Store generated outputs under a local artifact directory such as `artifacts/torch_osqp_stability/<timestamp>/`.
* Keep reproduction bundles next to the manifest and CSV.
* Do not commit generated evidence unless a release process explicitly requests it.
* Keep case data small enough for local reproduction and review.

---

## Required Methods

* `collect_stability_manifest(backends, settings, seeds)`.
* `run_stability_case(case, backend, settings)`.
* `write_case_results_csv(results, path)`.
* `write_manifest_json(manifest, path)`.
* `write_failure_reproduction(case, results, exception, path)`.
* `render_stability_summary(manifest, results)`.

---

## Validation Rules

* CSV rows must include every required field.
* Manifest must include commit, platform, hardware, Python, PyTorch, OSQP, backend, settings, and seeds.
* Every failed supported case must have a reproduction bundle.
* Summary counts must match CSV row counts.
* Supported size gate remains `n+m <= 2400`.
* Artifacts remain local unless publication is explicitly approved.

---

## UI / API Integration

* Provide a script or pytest command for local evidence generation.
* Emit artifact paths at the end of the run.
* Keep generated evidence out of normal imports.
* Keep the evidence runner backend-aware so support claims can be promoted independently.

---

## Workflow

1. Collect manifest metadata.
2. Generate deterministic case families and seed lists.
3. Run each case/backend bucket.
4. Write one CSV row per case/backend result.
5. Serialize reproduction data for every failure.
6. Render the Markdown summary.
7. Review support claims based on zero unexplained supported-case failures.

---

## Files to Create

* `scripts/torch_osqp_stability.py`: local evidence runner.
* `tests/test_torch_osqp_stability_artifacts.py`: artifact schema tests.

---

## Files to Modify

* `.gitignore`: ignore local stability artifact directories if not already covered.
* `docs/TORCH_OSQP_COMPLETION_AUDIT.md`: summarize evidence status and artifact paths.
* `docs/FULL_DEVELOPMENT_AND_VALIDATION_PIPELINE.md`: reference the evidence package command after implementation.

---

## Error Handling

* Continue running independent cases after a case failure.
* Mark unsupported backend/device combinations as skipped with reason.
* Fail the evidence command when a supported bucket has unexplained failures.
* Write partial artifacts when the run is interrupted after cases have completed.

---

## Testing Checklist

- [x] CSV schema contains all required columns.
- [x] Manifest schema contains environment, backend, settings, and seed data.
- [x] Failure reproduction bundle is written for every failure.
- [x] Markdown summary totals match CSV data.
- [x] Unsupported backend skips include explicit reasons.
- [x] Local artifact directory is ignored or clearly excluded from release commits.

---

## Acceptance Criteria

* Evidence generation produces the required CSV, JSON manifest, Markdown summary, and failure bundles.
* Artifacts remain local by default.
* Support promotion decisions can be traced to case-level evidence.
