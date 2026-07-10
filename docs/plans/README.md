# Torch-OSQP Plans

This directory contains tiny milestone plans for the current PyGRANSO
Torch-OSQP dense reference project.

The style follows `C:/Users/1/Downloads/phase_2.1_plan.md`: every file is a
planning blueprint with existing assets, data-structure design, ASCII-style
class/module details, implementation checkboxes, validation checkboxes, and exit
criteria.
The downloaded file is used only as a style model; its browser-extension content
is intentionally ignored.

No local `phase_2.1_plan.md` duplicate is created.

## Roadmap

- [roadmap.md](roadmap.md) — decision-complete roadmap with support envelope,
  architecture, lifecycle contracts, evidence gates, and verified/pending
  checkbox status.

## Tiny milestone plans

### Phase 0 — Research snapshot and active-path cleanup

- [phase_0.1_archive_snapshot_plan.md](phase_0.1_archive_snapshot_plan.md)
- [phase_0.2_remove_research_paths_plan.md](phase_0.2_remove_research_paths_plan.md)

### Phase 1 — Public contract and adapter policy

- [phase_1.1_public_qp_contract_plan.md](phase_1.1_public_qp_contract_plan.md)
- [phase_1.2_backend_policy_and_fallback_plan.md](phase_1.2_backend_policy_and_fallback_plan.md)
- [phase_1.3_settings_validation_and_migration_plan.md](phase_1.3_settings_validation_and_migration_plan.md)

### Phase 2 — Dense Torch solver internals

- [phase_2.1_dense_lu_workspace_plan.md](phase_2.1_dense_lu_workspace_plan.md)
- [phase_2.2_direct_admm_kernel_plan.md](phase_2.2_direct_admm_kernel_plan.md)
- [phase_2.3_scaling_adaptive_polishing_plan.md](phase_2.3_scaling_adaptive_polishing_plan.md)

### Phase 3 — PyGRANSO integration and builtin parity

- [phase_3.1_builtin_parity_plan.md](phase_3.1_builtin_parity_plan.md)
- [phase_3.2_pygranso_integration_plan.md](phase_3.2_pygranso_integration_plan.md)

### Phase 4 — Validation, evidence, and backend promotion

- [phase_4.1_tests_and_differential_plan.md](phase_4.1_tests_and_differential_plan.md)
- [phase_4.2_stability_evidence_plan.md](phase_4.2_stability_evidence_plan.md)
- [phase_4.3_platform_promotion_plan.md](phase_4.3_platform_promotion_plan.md)

### Phase 5 — Documentation and release handoff

- [phase_5.1_documentation_pdf_release_plan.md](phase_5.1_documentation_pdf_release_plan.md)
- [phase_5.2_fork_local_hardening_plan.md](phase_5.2_fork_local_hardening_plan.md)

## Planning conventions

- [x] Each plan uses current PyGRANSO/Torch-OSQP files and functions.
- [x] Each plan includes UML-style class diagrams for every class/data holder it names.
- [x] Each plan breaks work into small checkboxes.
- [x] Each plan separates implementation, validation, and exit criteria.
- [x] Plans describe future/review work and must not overclaim support evidence.
