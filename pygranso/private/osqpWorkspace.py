"""Optimizer-owned state for builtin and Torch OSQP routes."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from pygranso.private.torchQDLDLSolve import QDLDLSolver
from pygranso.private.torchSparseSolve import SparseDirectSolver


@dataclass
class TorchOSQPWorkspace:
    """State that must not be shared between independent PyGRANSO runs."""

    state: dict | None = None
    work_iterates: dict | None = None
    # solver->solution as last written by store_solution (x, y).
    public_solution: tuple | None = None
    problem_signature: tuple | None = None
    constraint_order_signature: tuple | None = None
    constraint_rows: torch.Tensor | None = None
    constraint_types: torch.Tensor | None = None
    p_pattern: torch.Tensor | None = None
    a_pattern: torch.Tensor | None = None
    scaling: dict | None = None
    scaling_source_p: torch.Tensor | None = None
    scaling_source_a: torch.Tensor | None = None
    scaling_passes: int = 0
    scaling_setting: int | None = None
    rho_bar: float | None = None
    rho_setting: float | None = None
    rho_setup_signature: tuple | None = None
    adaptive_rho_interval: int | None = None
    adaptive_rho_setup_time: float | None = None
    last_rel_kkt: float | None = None
    rho_updates_committed: int = field(default=0, init=False)
    status_polish: int = field(default=0, init=False)
    active_backend: str | None = None
    linear_solver: QDLDLSolver | SparseDirectSolver = field(default_factory=QDLDLSolver)
    last_info: dict | None = None
    status: str = "unsolved"
    status_problem: tuple | None = None
    builtin_cache: dict | None = None
    builtin_stats: dict = field(
        default_factory=lambda: {
            "setups": 0,
            "updates": 0,
            "rebuilds": 0,
            "last_cache_hit": False,
        }
    )

    def reset_torch(self) -> None:
        self.state = None
        self.work_iterates = None
        self.public_solution = None
        self.problem_signature = None
        self.constraint_order_signature = None
        self.constraint_rows = None
        self.constraint_types = None
        self.p_pattern = None
        self.a_pattern = None
        self.scaling = None
        self.scaling_source_p = None
        self.scaling_source_a = None
        self.scaling_passes = 0
        self.scaling_setting = None
        self.rho_bar = None
        self.rho_setting = None
        self.rho_setup_signature = None
        self.adaptive_rho_interval = None
        self.adaptive_rho_setup_time = None
        self.last_rel_kkt = None
        self.rho_updates_committed = 0
        self.status_polish = 0
        self.linear_solver.reset()
        self.last_info = None
        self.status = "unsolved"
        self.status_problem = None

    def reset_builtin(self) -> None:
        self.builtin_cache = None
        self.builtin_stats = {
            "setups": 0,
            "updates": 0,
            "rebuilds": 0,
            "last_cache_hit": False,
        }

    def reset(self) -> None:
        self.reset_torch()
        self.reset_builtin()
        self.active_backend = None

    def ensure_backend(self, backend: str) -> bool:
        """Invalidate all cached state when a run changes QP backend."""

        if backend not in {"builtin", "torch"}:
            raise ValueError(f"Unknown workspace backend {backend!r}.")
        changed = self.active_backend is not None and self.active_backend != backend
        if changed:
            self.reset_torch()
            self.reset_builtin()
        self.active_backend = backend
        return changed
