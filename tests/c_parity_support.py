"""Shared helpers for bit-level comparisons between pinned C OSQP and Torch."""

from __future__ import annotations

import numpy as np
import torch
from pinned_osqp_c_internals import csc_to_dense, read_workspace
from pinned_osqp_c_session import OSQP_INFTY, PinnedOSQPSession, paired_settings

from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchOSQP import solve_torch_osqp_direct

POLISH_CODES = {
    "accepted": 1,
    "rejected_no_improvement": -1,
    "skipped_no_active_constraints": 2,
    "disabled": 0,
    "not_run": 0,
    "skipped_unsolved": 0,
}

INFO_FIELDS = (
    ("status_val", "status_val"),
    ("iter", "admm_iterations"),
    ("rho_updates", "rho_updates"),
    ("rho", "rho_bar"),
    ("rho_estimate", "rho_estimate"),
    ("objective", "objective"),
    ("dual_objective", "dual_objective"),
    ("duality_gap", "duality_gap"),
    ("primal_residual", "primal_residual"),
    ("dual_residual", "dual_residual"),
    ("rel_kkt_error", "rel_kkt_error"),
)


def same_bits(a, b):
    """Compare binary64 bytes including the sign bit of zero."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    return a.tobytes() == b.tobytes()


def tensor(values):
    return torch.as_tensor(np.asarray(values, dtype=np.float64), dtype=torch.float64)


def torch_solve(problem, torch_settings, workspace):
    P, q, A, l, u = problem
    x, info = solve_torch_osqp_direct(
        tensor(P), tensor(q), tensor(A).reshape(len(l), len(q)), tensor(l), tensor(u),
        dict(torch_settings), workspace,
    )
    return x.reshape(-1).numpy(), info


def compare_solve(c_result, c_work, x, info, workspace):
    """Return the list of quantities that are NOT bit-identical after a solve."""

    bad = []
    for c_name, t_name in INFO_FIELDS:
        c_value, t_value = c_result[c_name], info[t_name]
        if isinstance(c_value, int):
            if int(c_value) != int(t_value):
                bad.append(f"{c_name}: C={c_value} T={t_value}")
        elif not same_bits(c_value, t_value):
            bad.append(f"{c_name}: C={c_value!r} T={t_value!r}")
    has_solution = c_result["status_val"] not in (3, 4, 5, 6, 9)
    if has_solution:
        if not same_bits(c_result["x"], x):
            bad.append("public x")
        if not same_bits(c_result["y"], info["state"]["y"].numpy()):
            bad.append("public y")
        retained = workspace.work_iterates
        for key in ("x", "z", "y"):
            if not same_bits(c_work[key], retained[key].numpy()):
                bad.append(f"work {key}")
    else:
        if not same_bits(c_result["x"], x):
            bad.append("public no-solution x")
        if workspace.work_iterates is not None:
            bad.append("iterates should be cold-started")
        if not all(np.all(c_work[key] == 0.0) for key in ("x", "z", "y")):
            bad.append("C iterates not cold-started")
        if c_result["status_val"] in (3, 4):
            cert = info["prim_inf_cert"]
            if cert is None or not same_bits(c_result["prim_inf_cert"], cert.numpy()):
                bad.append("primal infeasibility certificate")
        if c_result["status_val"] in (5, 6):
            cert = info["dual_inf_cert"]
            if cert is None or not same_bits(c_result["dual_inf_cert"], cert.numpy()):
                bad.append("dual infeasibility certificate")
    if info["status_polish"] != c_result["status_polish"]:
        bad.append(f"status_polish: C={c_result['status_polish']} T={info['status_polish']}")
    return bad


def compare_setup(c_work, workspace, torch_settings, problem):
    """Bit-level comparison of everything ``osqp_setup`` produces."""

    bad = []
    solver = workspace.linear_solver
    scaling = workspace.scaling
    n = c_work["n"]
    if torch_settings["scaling"]:
        for c_name, t_name in (("D", "D"), ("E", "E"), ("c", "cost")):
            if not same_bits(c_work[c_name], scaling[t_name].numpy()):
                bad.append(f"scaling {c_name}")
        if not same_bits(csc_to_dense(c_work["P"]), torch.triu(scaling["scaled_P"]).numpy()):
            bad.append("scaled P")
        if not same_bits(csc_to_dense(c_work["A"]), scaling["scaled_A"].numpy()):
            bad.append("scaled A")
    lin = c_work["linsys"]
    if solver.permutation != lin["perm"].tolist():
        bad.append("AMD permutation")
    Kp, Ki, Kx = solver.permuted_kkt
    if Kp != lin["KKT"]["p"].tolist() or Ki != lin["KKT"]["i"].tolist():
        bad.append("permuted KKT pattern")
    Lp, Li, Lx, D, Dinv = solver.factors
    if Lp != lin["L"]["p"].tolist() or Li != lin["L"]["i"].tolist():
        bad.append("L pattern")
    del n
    return bad


def compare_factors(c_work, workspace):
    """Bit-level comparison of the current numeric KKT values and LDL' factors."""

    bad = []
    solver = workspace.linear_solver
    lin = c_work["linsys"]
    _, _, Kx = solver.permuted_kkt
    _, _, Lx, D, Dinv = solver.factors
    for name, c_value, t_value in (
        ("permuted KKT values", lin["KKT"]["x"], Kx),
        ("L values", lin["L"]["x"], Lx),
        ("D", lin["D"], D),
        ("Dinv", lin["Dinv"], Dinv),
    ):
        if not same_bits(c_value, t_value):
            bad.append(name)
    return bad


def random_qp(seed, *, n=None, m=None, density=None, infeasible=None, unbounded=False):
    """A convex QP with mixed constraint types; ``+-1e30`` marks infinite bounds."""

    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 16)) if n is None else n
    m = int(rng.integers(0, 22)) if m is None else m
    density = float(rng.choice([0.2, 0.5, 1.0])) if density is None else density
    M = rng.standard_normal((n, n)) * (rng.random((n, n)) < density)
    P = (M @ M.T + 0.05 * np.eye(n)) * 10.0 ** rng.uniform(-1, 1)
    if unbounded:
        P = np.zeros((n, n))
    q = rng.standard_normal(n) * 10.0 ** rng.uniform(-1, 1)
    A = rng.standard_normal((m, n)) * (rng.random((m, n)) < max(density, 0.3)) + 0.0
    x0 = rng.standard_normal(n)
    Ax = A @ x0
    l = Ax - rng.uniform(0.1, 1.0, m)
    u = Ax + rng.uniform(0.1, 1.0, m)
    for i in range(m):
        r = rng.random()
        if r < 0.15:
            l[i] = u[i] = Ax[i]
        elif r < 0.3:
            l[i] = -OSQP_INFTY
        elif r < 0.45:
            u[i] = OSQP_INFTY
        elif r < 0.5:
            l[i], u[i] = -OSQP_INFTY, OSQP_INFTY
    if infeasible and m >= 2:
        A[-1] = A[-2]
        l[-2], u[-2] = Ax[-2] - 0.5, Ax[-2] + 0.5
        l[-1], u[-1] = u[-2] + 1.0, u[-2] + 1.5
    return P, q, A, l, u


__all__ = [
    "OSQP_INFTY", "PinnedOSQPSession", "TorchOSQPWorkspace", "compare_factors",
    "compare_setup", "compare_solve", "paired_settings", "random_qp",
    "read_workspace", "same_bits", "tensor", "torch_solve",
]
