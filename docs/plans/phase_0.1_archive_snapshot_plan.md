# Phase 0.1 Implementation Plan: Research Snapshot Archive

## Goal

Implement **research snapshot archive verification**.

This feature should allow the project to:

1. Prove the sparse-CG/CUDA Graph research line is recoverable.
2. Record branch, tag, signature, and baseline validation evidence.
3. Let mainline dense Torch-OSQP work proceed without losing prior research.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

---

## Current State

The project already has:

* `docs/TORCH_OSQP_COMPLETION_AUDIT.md`: records archive branch and tag evidence.
* `.codex/code-edit-log.md`: records work-session evidence and baseline notes.
* Git refs: expected archive branch `archive/sparse-cg-cuda-graph` and tag `research-sparse-cg-cuda-graph-final`.

The missing part is:

* A standalone checklist plan for verifying the archive evidence.
* A clear data model for archive proof records.
* A repeatable validation workflow for future auditors.

---

## New Components to Add

Add the following planning components.

### Component 1

```text
ArchiveEvidenceRecord
```

Responsibility:

```text
Track branch, tag, baseline validation, and recovery instructions as one reviewable archive proof.
```

### Component 2

```text
GitRefVerifier
```

Responsibility:

```text
Verify local and remote refs resolve to the expected archive commit and expose clear failure messages.
```

---

## Class / Registry Diagrams

### Diagram 1: Archive Evidence Record

```text
+-------------------------------------------------------------------------------+
|                              ArchiveEvidenceRecord                             |
+-------------------------------------------------------------------------------+
|  - archiveBranch: string                                                       |
|  - archiveTag: string                                                          |
|  - expectedCommit: string                                                      |
|  - baselineCommand: string                                                     |
|  - baselineResult: string                                                      |
|  - recoveryNotes: string[]                                                     |
+-------------------------------------------------------------------------------+
|  + summarize(): string                    --> Returns reviewer-facing summary  |
|  + isComplete(): boolean                  --> Checks required evidence fields  |
|  + missingFields(): string[]              --> Lists absent evidence fields     |
+-------------------------------------------------------------------------------+
```

### Diagram 2: Git Ref Verifier

```text
+-------------------------------------------------------------------------------+
|                                  GitRefVerifier                                |
+-------------------------------------------------------------------------------+
|  - remoteName: string                                                          |
|  - localRefs: Map<string, string>                                              |
|  - remoteRefs: Map<string, string>                                             |
+-------------------------------------------------------------------------------+
|  + verifyLocal(ref): boolean              --> Confirms local ref resolves      |
|  + verifyRemote(ref): boolean             --> Confirms remote ref resolves     |
|  + verifyTarget(ref, sha): boolean        --> Confirms ref target commit       |
|  + verifyTagSignature(tag): boolean       --> Checks signed tag if configured  |
+-------------------------------------------------------------------------------+
```

### Diagram 3: Validation Snapshot

```text
+-------------------------------------------------------------------------------+
|                                ValidationSnapshot                              |
+-------------------------------------------------------------------------------+
|  - command: string                                                             |
|  - passedCount: number                                                         |
|  - failedCount: number                                                         |
|  - artifactPath: string | null                                                 |
|  - notes: string                                                               |
+-------------------------------------------------------------------------------+
|  + isAcceptableBaseline(): boolean        --> Determines archive baseline use  |
|  + toAuditText(): string                  --> Formats audit-log text           |
+-------------------------------------------------------------------------------+
```

---

## Class Diagram Rules

* Treat the archive branch and signed tag as release-preservation artifacts, not runtime classes.
* Keep snapshot metadata separate from implementation cleanup work.
* Do not require local signing inside this phase; the human will configure signing separately.

---

## Data Model

```ts
type ArchiveEvidence = {
  id: string;
  name: string;
  archiveBranch: string;
  archiveTag: string;
  expectedCommit: string;
  baselineCommand?: string;
  baselineResult?: string;
  recoveryNotes?: string[];
  createdAt?: string;
  updatedAt?: string;
};
```

Required fields:

* `id`
* `name`
* `archiveBranch`
* `archiveTag`
* `expectedCommit`

---

## Storage / State

This feature uses persistent project documentation, not runtime state.

```text
Storage key: not applicable
Storage location: docs/TORCH_OSQP_COMPLETION_AUDIT.md and .codex/code-edit-log.md
```

---

## Required Methods

Use function-style verification:

```ts
function verifyArchiveEvidence(input: ArchiveEvidence): VerificationReport
```

Expected output:

```ts
type VerificationReport = {
  success: boolean;
  missing?: string[];
  errors?: string[];
  summary: string;
};
```

---

## Validation Rules

Before accepting archive evidence, check:

1. Archive branch name is present.
2. Archive tag name is present.
3. Local branch resolves.
4. Remote branch resolves.
5. Tag resolves.
6. Signed-tag status is either verified or explicitly documented as externally configured.
7. Baseline result is recorded.

---

## UI / API Integration

This feature has no UI or API surface.

Internal callers:

* Release auditor reads the audit and edit log.
* Codex or a maintainer runs git verification commands.

---

## Workflow

1. Read expected branch and tag names from the audit.
2. Verify local refs.
3. Verify remote refs.
4. Verify tag metadata.
5. Verify baseline validation notes.
6. Record any missing evidence.
7. Update audit or edit log only if evidence changes.

---

## Files to Create

```text
None
```

Only create new files if archive evidence needs a dedicated manifest later.

---

## Files to Modify

```text
docs/TORCH_OSQP_COMPLETION_AUDIT.md
.codex/code-edit-log.md
```

Modify only if archive evidence or recovery instructions change.

---

## Error Handling

Handle these cases:

* Local archive branch is missing.
* Remote archive branch is missing.
* Tag is missing.
* Tag is unsigned or signing cannot be verified.
* Baseline result is absent.
* Expected commit differs from actual ref target.

Prefer clear audit notes over silent assumptions.

---

## Testing Checklist

Test the following:

* [ ] Local branch resolves.
* [ ] Remote branch resolves.
* [ ] Tag resolves.
* [ ] Tag signature status is documented.
* [ ] Baseline validation is recorded.
* [ ] Recovery path is clear.
* [ ] Active mainline work does not depend on archive-only code.

---

## Acceptance Criteria

This phase is complete when:

1. `ArchiveEvidenceRecord` is complete.
2. Branch and tag evidence are verified or explicitly documented.
3. Baseline validation evidence is preserved.
4. Recovery instructions are readable.
5. The archive proof does not overclaim active solver support.
