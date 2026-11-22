"""Float64 regressions for solve retry and termination check ordering."""

import pytest
import torch
from pinned_osqp_c_oracle import paired_settings

from pygranso.private import torchOSQP as solver
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace


def _timeout_problem():
    P = torch.ones((1, 1), dtype=torch.float64)
    return P, -P.new_ones(1), P.clone(), P.new_zeros(1), P.new_full((1,), 0.25)


def _timeout_settings(time_limit):
    _, settings = paired_settings(
        rho=0.125, sigma=0.125, alpha=1.0, eps_abs=1e-6, eps_rel=1e-6,
        scaling=0, max_iter=1, check_termination=4, check_dualgap=0,
        adaptive_rho=2, adaptive_rho_interval=0, adaptive_rho_fraction=0.25,
        adaptive_rho_tolerance=1e12, warm_starting=0,
    )
    return settings | {"time_limit": time_limit}


def _solve_with_clock(monkeypatch, problem, settings, workspace, ticks):
    clock = iter(ticks)
    decisions = []
    terminate = solver._termination_status

    def observe(*args, **kwargs):
        result = terminate(*args, **kwargs)
        decisions.append((kwargs["approximate"], result))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(solver, "_clock", lambda: next(clock))
        patch.setattr(solver, "_termination_status", observe)
        _, info = solver.solve_torch_osqp_direct(*problem, settings, workspace)
    assert next(clock, None) is None
    return info, decisions


@pytest.mark.parametrize("updated", [False, True], ids=["setup", "changed-q"])
def test_native_timeout_status_survives_compatible_retry(monkeypatch, updated):
    # Native status sequence and clock schedule recorded from pinned C.
    problem = _timeout_problem()
    workspace = TorchOSQPWorkspace()
    settings = _timeout_settings(3.0 if updated else 2.75)
    if updated:
        before, _ = _solve_with_clock(
            monkeypatch, problem, settings, workspace, [-2, 0, 0.5, 0.5],
        )
        assert before["status_val"] == 7
        problem[1].fill_(-1.25)
    timeout, _ = _solve_with_clock(
        monkeypatch, problem, settings, workspace, [-2.25 if updated else -2, 0, 0.75],
    )
    retained = {key: value.clone() for key, value in workspace.state.items()}
    retry, decisions = _solve_with_clock(
        monkeypatch, problem, settings, workspace, [0, 0, 0.75, 0.75],
    )

    assert timeout["status_val"] == retry["status_val"] == 8
    assert timeout["admm_iterations"] == 0
    assert retry["admm_iterations"] == retry["linear_solves_this_solve"] == 1
    assert retry["factorizations_this_solve"] == retry["rho_updates"] == 0
    assert retry["rho_bar"] == 0.125
    assert decisions == [(False, None), (True, None)]
    for key, value in retained.items():
        assert torch.equal(workspace.state[key], value)


def test_native_prior_iteration_cap_skips_approximate_retry(monkeypatch):
    workspace = TorchOSQPWorkspace()
    problem = _timeout_problem()
    settings = _timeout_settings(3.0)
    first, _ = _solve_with_clock(
        monkeypatch, problem, settings, workspace, [-2, 0, 0.75, 0.75],
    )
    retry, decisions = _solve_with_clock(
        monkeypatch, problem, settings, workspace, [0, 0, 0.75],
    )
    assert first["status_val"] == retry["status_val"] == 7
    assert decisions == [(False, None)]


@pytest.mark.parametrize(
    "max_iter,interval,exact_calls,metric_calls",
    [(325, 25, 13, 14), (326, 25, 14, 16), (325, 0, 1, 3)],
    ids=["checked-cap", "unchecked-cap", "periodic-disabled"],
)
def test_native_footer_check_order(monkeypatch, max_iter, interval, exact_calls, metric_calls):
    # Check and refresh counts recorded from pinned C at the same natural caps.
    P = torch.tensor([[4.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    q = P.new_ones(2)
    A = P.new_tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    l, u = P.new_tensor([1.0, 0.0, 0.0, -1e30]), P.new_tensor([1.0, 0.7, 0.7, 1e30])
    _, settings = paired_settings(
        scaling=0, adaptive_rho=0, warm_starting=0, eps_abs=1e-8, eps_rel=1e-8,
        max_iter=max_iter, check_termination=interval,
    )
    # Use the same effectively unlimited solve time as the archived fixture.
    settings["time_limit"] = 1e10
    terminate, residuals = solver._termination_status, solver._residuals
    decisions = []
    metric_updates = 0

    def observe_termination(*args, **kwargs):
        result = terminate(*args, **kwargs)
        decisions.append((kwargs["approximate"], result))
        return result

    def observe_residuals(P, q, A, x, z, y, eps_abs, eps_rel, **kwargs):
        nonlocal metric_updates
        # Final rho estimation requests zero tolerances.
        if eps_abs != 0.0:
            metric_updates += 1
        return residuals(P, q, A, x, z, y, eps_abs, eps_rel, **kwargs)

    monkeypatch.setattr(solver, "_termination_status", observe_termination)
    monkeypatch.setattr(solver, "_residuals", observe_residuals)
    _, info = solver.solve_torch_osqp_direct(P, q, A, l, u, settings)

    assert decisions[:-1] == [(False, None)] * exact_calls
    approximate, result = decisions[-1]
    assert approximate and result[0] == "solved_inaccurate"
    assert metric_updates == metric_calls
    assert (info["status_val"], info["admm_iterations"]) == (2, max_iter)
    assert info["rho_updates"] == 0
    assert info["factorizations_this_solve"] == 1
