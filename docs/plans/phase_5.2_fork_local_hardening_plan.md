# Phase 5.2 Implementation Plan: Fork-Local Release Candidate Hardening

## Goal

Implement **fork-local release candidate hardening**.

This step should allow us to:

1. Treat the fork, not the original repo PR, as the active validation target.
2. Cleanly separate “project quality is good enough” from “upstream PR is ready.”
3. Verify the forked project through repeatable local/fork evidence before any future PR decision.

Keep this documentation/test-focused, conservative, and consistent with the current roadmap.

## Current State

The project already has:

- `docs/plans/roadmap.md`: main checkbox/status source.
- `docs/TORCH_OSQP_COMPLETION_AUDIT.md`: current release audit.
- `.github/workflows/torch-osqp-core.yml`: deterministic core CI.
- `.github/workflows/torch-osqp-nightly.yml`: CPU nightly stability gate.
- `.github/workflows/torch-osqp-cuda-promotion.yml`: CUDA promotion gate, still unpromoted.
- `scripts/render_pipeline_pdf.py`: pipeline PDF renderer.
- `tests/`: deterministic, OSQP, PyGRANSO, metamorphic, randomized, and reporting tests.
- `F:\UMN Researches\Ju Research\Report`: external research notebook and artifact storage.

Current evidence:

- Fork `main` core CI passed.
- Fork `main` nightly CPU stability passed.
- 1800 CPU cases, 0 release-gate failures.
- CUDA remains unpromoted.
- ROCm/MPS remain unclaimed.
- Local full pytest is blocked by a local temp-directory permission issue, but 71 tests passed and the affected reporting behavior was separately smoke-tested.

The missing part is:

- Roadmap/audit wording still frames upstream PR/review as an active release action.
- We need a fork-local “good enough” checklist.
- We need one more clean local/fork release-candidate review that does not depend on the original repo.

## New Components to Add

No new production solver components should be added in this phase.

Optional only:

### Component 1

`ForkReleaseCandidateChecklist`

Responsibility:

A lightweight checklist, likely documented rather than coded, that records fork-local readiness checks, evidence locations, known limitations, and final “do not promote accelerators yet” status.

Skip this component if the existing `Report` tracker is enough.

## Class / Registry Diagrams

### Stateless Utility / Checklist

```text
+-------------------------------------------------------------------------------+
|                         ForkReleaseCandidateChecklist                          |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + inspect_repo_state(): Report              --> Confirms clean local branch    |
|  + verify_fork_ci(): Report                  --> Checks fork core/nightly runs  |
|  + verify_artifacts(): Report                --> Aggregates CSV/manifest data   |
|  + run_local_smoke_tests(): Report           --> Runs meaningful local tests    |
|  + classify_blockers(): Report               --> Separates local/fork/upstream  |
+-------------------------------------------------------------------------------+
```

## Class Diagram Rules

1. Do not add production classes for this phase.
2. Keep this phase documentation/test-only unless an actual repeated manual check needs scripting.
3. If a helper script is added later, make it stateless.
4. Do not expose any user-facing API.
5. Do not modify solver math or backend behavior.

## Data Model

This phase does not need persistent project data.

Use a simple evidence record shape in the Report notebook:

```python
ReleaseCandidateEvidence = {
    "repo": str,
    "branch": str,
    "commit": str,
    "core_run": str,
    "nightly_run": str,
    "total_cases": int,
    "release_gate_failures": int,
    "known_limitations": list[str],
    "next_required_human_action": str,
}
```

## Storage / State

Temporary / external state only.

- Repo files: only modify if roadmap/audit wording needs to reflect fork-local strategy.
- External notebook: continue using `F:\UMN Researches\Ju Research\Report`.
- Artifacts: keep local in `Report`, not committed.

## Required Methods

No code methods required now.

Operational commands/checks:

```powershell
git status --short --branch
gh run view <fork-core-run> --repo Ztang-Yit-Xiaang/PyGRANSO
gh run view <fork-nightly-run> --repo Ztang-Yit-Xiaang/PyGRANSO
python -m pytest tests -q
python scripts/render_pipeline_pdf.py   # only if docs change
```

If local pytest temp permissions fail again, record it and use:

```powershell
python -m pytest tests -q -p no:cacheprovider -k "not test_time_limit_exit_writes_partial_artifacts"
```

plus the direct reporting smoke test.

## Validation Rules

Before marking this phase done:

1. Fork `main` must be the validation target.
2. No upstream/original PR work should be performed.
3. CPU evidence must show 0 release-gate failures.
4. Float64 remains authoritative.
5. Float32 remains qualified.
6. CUDA remains unpromoted.
7. ROCm/MPS remain unclaimed.
8. Local test limitations must be classified as environment issues only when CI evidence covers the same logic.
9. Any roadmap/audit edit must avoid marking future or hardware-gated items complete.

## UI / API Integration

No UI/API integration.

Internal callers are human/Codex release workflow steps:

- Input: repo state, fork CI runs, artifact CSVs/manifests, roadmap/audit files.
- Output: local readiness summary and clear next blocker classification.
- Errors: missing artifacts, failed fork CI, unsupported accelerator evidence, dirty repo, local temp permission limitations.

## Workflow

1. Inspect local repo cleanliness.
2. Inspect fork `main` status and latest validation runs.
3. Aggregate latest fork-main nightly artifacts.
4. Run local deterministic tests where meaningful.
5. Classify local failures as either:
   - real project blocker,
   - local environment issue,
   - non-gating stress evidence.
6. Review roadmap/audit wording.
7. If docs still over-focus on upstream PR, revise wording to say upstream is future handoff, not current active path.
8. Regenerate PDF only if the main pipeline doc changes.
9. Update `.codex/code-edit-log.md` if repo files change.
10. Update external Report tracker.
11. Stop before any original/upstream PR action.

## Files to Create

Only if useful:

- `F:\UMN Researches\Ju Research\Report\YYYY-MM-DD_fork_release_candidate_readiness.md`

Do not create new repo files unless we decide the roadmap/audit needs strategy clarification.

## Files to Modify

Possibly:

- `docs/plans/roadmap.md`
- `docs/TORCH_OSQP_COMPLETION_AUDIT.md`
- `.codex/code-edit-log.md`

Do not modify solver source in this phase.

## Error Handling

Handle:

- Fork CI failed: classify as blocker and inspect logs.
- Missing artifacts: rerun/download fork nightly.
- Local pytest temp issue: document environment limitation, use fork CI as authoritative, run partial local coverage.
- Dirty repo: inspect before any edit.
- Accelerator evidence missing: keep CUDA/ROCm/MPS unpromoted/unclaimed.
- Original repo PR temptation: skip; not in scope.

## Testing Checklist

- [x] `git status --short --branch` is clean before and after.
- [x] Fork `main` core run is still successful.
- [x] Fork `main` nightly run is still successful.
- [x] Latest artifact CSVs aggregate to 1800 cases and 0 release-gate failures.
- [x] Manifests show clean source, no timeout, no partial results.
- [x] Local deterministic tests are attempted.
- [x] Local environment-only failures are documented clearly.
- [x] Roadmap/audit do not overclaim accelerator support.
- [x] No original/upstream PR action is performed.
- [x] Report tracker is updated.

## Roadmap / Full Pipeline Update

If implementing this phase changes repo docs:

- Update roadmap/audit wording minimally.
- Do not check upstream merge/register tasks.
- Do not check CUDA/ROCm/MPS promotion tasks.
- Add a note that fork-local readiness is the active path before any future PR decision.

## Acceptance Criteria

This phase is complete when:

1. Fork-local release readiness is clearly defined.
2. Fork `main` evidence is verified and summarized.
3. Local test limitations are classified accurately.
4. Roadmap/audit wording matches your instruction: no active original repo PR work.
5. No accelerator support is overclaimed.
6. External Report tracker has the final checkpoint.
7. The repo remains clean.
8. The next blocker is either a real technical failure or a human decision to proceed toward PR later.
