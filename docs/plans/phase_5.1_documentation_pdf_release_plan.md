# Phase 5.1 Implementation Plan: Documentation, PDF, and Release Evidence

## Goal

Implement **the final documentation, PDF, audit, and release-evidence update**.

This feature should allow the system to:

1. Maintain a two-part document: reviewer executive summary plus decision-complete engineering specification.
2. Preserve mathematically correct KKT, ADMM, projection, dual-update, and residual equations.
3. Attach support matrix, risk table, decision log, evidence links, and code-edit reports to release work.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

---

## Current State

The project already has:

* `docs/FULL_DEVELOPMENT_AND_VALIDATION_PIPELINE.md`.
* `docs/TORCH_OSQP_COMPLETION_AUDIT.md`.
* Related documentation in `UNCONSTRAINED_AND_OSQP.md`, `MIXED_PRECISION.md`, and `TORCH_COMPILE.md`.
* Local generated plans under `docs/plans/`.

The missing part is:

* A final release-ready documentation pass after implementation evidence exists.
* A reproducible PDF generation path with controlled page breaks for code blocks and diagrams.
* Cross-links from plans, audit, evidence, and code-edit reports.

---

## New Components to Add

### Component 1

```text
DocumentationSourceSet
```

Responsibility:

```text
Track source Markdown documents that define the release narrative, engineering decisions, and support claims.
```

### Component 2

```text
RenderedPDFArtifact
```

Responsibility:

```text
Represent the generated PDF, including source revision, render command, output path, and visual checks.
```

### Component 3

```text
CompletionAuditEntry
```

Responsibility:

```text
Record implementation status, validation evidence, residual risks, and support-matrix decisions.
```

### Component 4

```text
CodeEditReportEntry
```

Responsibility:

```text
Record every code or code-adjacent project change with timing, validation, findings, and required human action.
```

---

## Class / Registry Diagrams

### Diagram 1: Documentation Source Set

```text
+-------------------------------------------------------------------------------+
|                              DocumentationSourceSet                            |
+-------------------------------------------------------------------------------+
|  - pipelineDoc: path                                                           |
|  - auditDoc: path                                                              |
|  - relatedDocs: list[path]                                                     |
|  - planDirectory: path                                                         |
+-------------------------------------------------------------------------------+
|  + validate_links(): LinkReport          --> local doc cross-link check        |
|  + validate_claims(evidence): ClaimReport --> support claims match evidence    |
|  + release_index(): dict                 --> docs included in release review   |
+-------------------------------------------------------------------------------+
```

### Diagram 2: Rendered PDF Artifact

```text
+-------------------------------------------------------------------------------+
|                              RenderedPDFArtifact                               |
+-------------------------------------------------------------------------------+
|  - sourcePath: path                                                            |
|  - outputPath: path                                                            |
|  - renderCommand: string                                                       |
|  - generatedAt: datetime                                                       |
|  - pageCount: int                                                              |
+-------------------------------------------------------------------------------+
|  + render(): path                        --> generates local PDF               |
|  + inspect_layout(): LayoutReport        --> page breaks/code block checks     |
|  + checksum(): string                    --> release artifact identity         |
+-------------------------------------------------------------------------------+
```

### Diagram 3: Completion Audit Entry

```text
+-------------------------------------------------------------------------------+
|                              CompletionAuditEntry                              |
+-------------------------------------------------------------------------------+
|  - phase: string                                                               |
|  - status: string                                                              |
|  - evidencePaths: list[path]                                                   |
|  - risks: list[string]                                                         |
|  - decisions: list[string]                                                     |
+-------------------------------------------------------------------------------+
|  + update_from_phase(phase): None       --> records implementation status      |
|  + verify_evidence_exists(): boolean    --> checks local artifact paths        |
|  + render_summary(): string             --> audit Markdown section             |
+-------------------------------------------------------------------------------+
```

### Diagram 4: Code Edit Report Entry

```text
+-------------------------------------------------------------------------------+
|                                CodeEditReportEntry                             |
+-------------------------------------------------------------------------------+
|  - goal: string                                                                |
|  - timing: string                                                              |
|  - changedFiles: list[path]                                                    |
|  - findings: list[string]                                                      |
|  - validation: list[string]                                                    |
|  - humanAction: list[string]                                                   |
+-------------------------------------------------------------------------------+
|  + append(logPath): None                --> repository work-session memory     |
|  + summarize_for_release(): string      --> release-note input                 |
+-------------------------------------------------------------------------------+
```

---

## Class Diagram Rules

* Documentation source files remain Markdown-first.
* The generated PDF is an artifact, not the source of truth.
* Support claims must be evidence-backed.
* Code-edit reports remain separate from required human action and reflection.

---

## Data Model

```text
ReleaseDocumentationIndex
  version: string
  date: date
  pipeline_doc: path
  audit_doc: path
  support_matrix: path
  stability_artifacts: list[path]
  pdf_artifact: path
  code_edit_log: path
```

```text
LayoutReport
  page_count: int
  broken_code_blocks: int
  broken_diagrams: int
  missing_toc: bool
  missing_page_breaks: list[string]
```

---

## Storage / State

* Store source documentation in `docs/`.
* Store phase plans in `docs/plans/`.
* Store local PDF/evidence outputs in a local artifact directory unless release policy approves committing them.
* Store code-edit reports in `.codex/code-edit-log.md`.

---

## Required Methods

* `validate_documentation_links(docs_dir)`.
* `validate_support_claims_against_evidence(audit_doc, evidence_dir)`.
* `render_pipeline_pdf(source, output)`.
* `inspect_pdf_layout(pdf_path)`.
* `update_completion_audit(phase, status, evidence)`.
* `append_code_edit_report(entry)`.

---

## Validation Rules

* The final pipeline document must include version/date, table of contents, support matrix, risk table, decision log, and controlled page breaks.
* Torch-direct must be described as a dense reference implementation, not a scalable sparse solver.
* Infeasibility certificates, nonconvex detection, and sparse acceleration remain later milestones.
* Public behavior must preserve `osqp_algebra={"auto","builtin","torch"}`.
* PDF layout must be visually checked when generated.
* Every implementation task that modifies project files must append a code-edit report.

---

## UI / API Integration

* No runtime API changes are expected in this documentation phase.
* Documentation must describe actual runtime behavior from earlier phases.
* Release notes should link to local evidence summaries when available.

---

## Workflow

1. Confirm implementation and evidence status for each phase.
2. Update the completion audit with evidence paths, risks, and decisions.
3. Update the full pipeline document with final support claims and release gates.
4. Validate local documentation links.
5. Render the PDF locally.
6. Inspect PDF layout for table of contents, code blocks, diagrams, and page breaks.
7. Write release summary and code-edit report entries.
8. Keep generated artifacts local unless the human approves publication.

---

## Files to Create

* `scripts/render_torch_osqp_pipeline_pdf.py`: optional reproducible PDF rendering helper if no existing doc build path fits.
* `tests/test_torch_osqp_docs_links.py`: optional documentation link and claim checks.

---

## Files to Modify

* `docs/FULL_DEVELOPMENT_AND_VALIDATION_PIPELINE.md`: final engineering specification and reviewer summary.
* `docs/TORCH_OSQP_COMPLETION_AUDIT.md`: phase status, evidence, risk, and decision updates.
* `docs/plans/README.md`: release-phase references if the plan index changes.
* `.codex/code-edit-log.md`: append code-edit reports for documentation and implementation edits.

---

## Error Handling

* Missing evidence blocks support claims.
* Broken local documentation links block release documentation completion.
* PDF rendering failure leaves Markdown as source of truth and records the issue in the audit.
* Layout failures require doc edits before the PDF is considered release-ready.

---

## Testing Checklist

- [x] Full pipeline document has reviewer summary and engineering specification.
- [x] KKT, ADMM, projection, dual-update, and residual equations are preserved.
- [x] Support matrix matches backend evidence.
- [x] Risk table and decision log are current.
- [x] Local plan links resolve.
- [x] PDF renders successfully when requested.
- [x] PDF layout is visually checked for code blocks, diagrams, and page breaks.
- [x] Code-edit report exists for every implementation task that modifies files.

---

## Acceptance Criteria

* Documentation accurately reflects the implemented dense Torch reference solver.
* Release support claims are evidence-backed and backend-specific.
* Generated artifacts remain local unless explicitly approved.
* The completion audit and code-edit log provide a clear release trail.
