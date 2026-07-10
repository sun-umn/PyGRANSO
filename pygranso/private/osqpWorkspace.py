"""Optimizer-owned state for builtin and Torch OSQP routes."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from pygranso.private.torchLinearSolve import DenseLUSolver


@dataclass
class TorchOSQPWorkspace:
    """State that must not be shared between independent PyGRANSO runs."""

    state: dict | None = None
    problem_signature: tuple | None = None
    constraint_order_signature: tuple | None = None
    p_pattern: torch.Tensor | None = None
    a_pattern: torch.Tensor | None = None
    scaling: dict | None = None
    scaling_source_p: torch.Tensor | None = None
    scaling_source_a: torch.Tensor | None = None
    scaling_passes: int = 0
    rho_bar: float | None = None
    rho_setting: float | None = None
    active_backend: str | None = None
    linear_solver: DenseLUSolver = field(default_factory=DenseLUSolver)
    last_info: dict | None = None
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
        self.problem_signature = None
        self.constraint_order_signature = None
        self.p_pattern = None
        self.a_pattern = None
        self.scaling = None
        self.scaling_source_p = None
        self.scaling_source_a = None
        self.scaling_passes = 0
        self.rho_bar = None
        self.rho_setting = None
        self.linear_solver.clear()
        self.last_info = None

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
