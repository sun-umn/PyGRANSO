# Phase 0.2 Implementation Plan: Remove Research Execution Paths

## Goal

Implement **active-path cleanup for removed sparse-CG/CUDA Graph research code**.

This feature should allow the system to:

1. Keep only builtin OSQP and dense Torch OSQP in the active package path.
2. Reject legacy research settings with actionable migration errors.
3. Preserve the research implementation only on the archive branch.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

---

## Current State

The project already has:

* `pygranso/private/osqpTorchAdapter.py`: contains `LEGACY_TORCH_SETTINGS` and backend policy.
* `docs/TORCH_OSQP_COMPLETION_AUDIT.md`: records removed research execution paths.
* `archive/sparse-cg-cuda-graph`: expected location of preserved research code.

The missing part is:

* A repeatable plan to confirm research execution does not re-enter active code.
* A clear data model for legacy setting rejection.
* Explicit tests/search checks that distinguish archived code from active runtime code.

---

## New Components to Add

### Component 1

```text
LegacySettingGuard
```

Responsibility:

```text
Detect archived sparse-CG/CUDA Graph options and raise migration errors before backend selection.
```

### Component 2

```text
ActivePathAudit
```

Responsibility:

```text
Search active package files for removed execution paths and summarize whether runtime cleanup remains true.
```

---

## Class / Registry Diagrams

### Diagram 1: Legacy Setting Guard

```text
+-------------------------------------------------------------------------------+
|                                LegacySettingGuard                              |
+-------------------------------------------------------------------------------+
|  - legacyKeys: Set<string>                                                     |
|  - migrationRelease: string                                                    |
|  - removalRelease: string                                                      |
+-------------------------------------------------------------------------------+
|  + detect(settings): string[]             --> Returns legacy keys in settings  |
|  + buildMessage(keys): string             --> Creates actionable error text    |
|  + raiseIfPresent(settings): void         --> Stops archived option execution  |
+-------------------------------------------------------------------------------+
```

### Diagram 2: Active Path Audit

```text
+-------------------------------------------------------------------------------+
|                                  ActivePathAudit                               |
+-------------------------------------------------------------------------------+
|  - packageRoots: string[]                                                      |
|  - forbiddenSymbols: string[]                                                  |
|  - archiveRef: string                                                          |
+-------------------------------------------------------------------------------+
|  + scan(): AuditFinding[]                 --> Searches active package files    |
|  + hasRuntimeReference(): boolean         --> True if forbidden code remains   |
|  + summarize(): string                    --> Produces audit-ready summary     |
+-------------------------------------------------------------------------------+
```

### Diagram 3: Removed Research Feature

```text
+-------------------------------------------------------------------------------+
|                               RemovedResearchFeature                           |
+-------------------------------------------------------------------------------+
|  - name: string                                                                |
|  - oldOption: string                                                           |
|  - archiveLocation: string                                                     |
|  - replacement: string                                                         |
|  - removalReason: string                                                       |
+-------------------------------------------------------------------------------+
|  + migrationNote(): string                --> Explains replacement path        |
|  + isArchivedOnly(): boolean              --> Confirms active path exclusion   |
+-------------------------------------------------------------------------------+
```

---

## Class Diagram Rules

* Treat removed research paths as implementation boundaries, not user-facing features.
* Keep archival metadata out of the main package path.
* Preserve dense direct behavior while deleting CG, sparse-operator, CUDA Graph, and npm artifacts.

---

## Data Model

```ts
type RemovedResearchFeature = {
  id: string;
  name: string;
  oldOption: string;
  archiveLocation: string;
  replacement: string;
  removalReason?: string;
  createdAt?: string;
  updatedAt?: string;
};
```

---

## Storage / State

This feature is mostly stateless. It receives settings or source paths, returns
validation/audit output, and does not persist runtime data.

Persistent documentation state lives in:

```text
docs/TORCH_OSQP_COMPLETION_AUDIT.md
docs/FULL_DEVELOPMENT_AND_VALIDATION_PIPELINE.md
```

---

## Required Methods

```ts
function detectLegacySettings(settings: dict): string[]
```

```ts
function auditActiveResearchPaths(packageRoots: string[]): AuditFinding[]
```

---

## Validation Rules

Before backend selection:

1. Detect `linear_solver`, `cg_*`, `cuda_graph`, and archived auto-selection settings.
2. Reject legacy settings with migration guidance.
3. Do not silently ignore settings that used to change solver behavior.
4. Do not import removed research modules from active code.

---

## UI / API Integration

This feature has no UI.

API surface:

* `_normalize_options(options, dtype)` calls the legacy-setting guard.
* Tests call the adapter and assert migration errors.

---

## Workflow

1. User passes `osqp_settings`.
2. Adapter normalizes options.
3. Legacy guard detects archived keys.
4. Adapter raises an actionable error.
5. Active-path audit confirms removed runtime paths are absent.

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
docs/TORCH_OSQP_COMPLETION_AUDIT.md
```

---

## Error Handling

Handle these cases:

* User requests archived CG settings.
* User requests CUDA Graph settings.
* User requests sparse-operator auto-selection settings.
* Active code accidentally imports archive-only modules.

---

## Testing Checklist

Test the following:

* [ ] Legacy `linear_solver` setting raises.
* [ ] Legacy CG tolerance setting raises.
* [ ] Legacy CUDA Graph setting raises.
* [ ] Active package path has no custom CG execution.
* [ ] Active package path has no CUDA Graph execution.
* [ ] Archive branch remains the recovery path.

---

## Acceptance Criteria

This phase is complete when:

1. `LegacySettingGuard` behavior is active.
2. Removed research options cannot execute active code.
3. Errors are clear and safe.
4. Tests or search checks confirm the active path is dense-reference only.
