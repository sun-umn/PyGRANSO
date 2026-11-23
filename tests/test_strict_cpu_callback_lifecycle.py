"""Retained native failure and repair sequences for the strict direct/adapter split."""

import numpy as np
import pytest
import torch
from pinned_osqp_c_failure_reference import adapt, matrix, native, polish, rho, step
from pinned_osqp_c_oracle import pinned_dll_path
from pinned_osqp_c_session import paired_settings

from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.solveQP import getErr, getLastOSQPInfo, solveQP
from pygranso.private.torchLinearSolve import TorchLinearSolveError
from pygranso.private.torchOSQP import solve_torch_osqp_direct
from pygranso.private.torchQDLDLSolve import QDLDLSolver

needs_dll = pytest.mark.skipif(
    not pinned_dll_path().is_file(), reason="pinned OSQP C library is unavailable"
)


def _settings(**changes):
    _, settings = paired_settings(
        **(
            {"scaling": 0, "adaptive_rho": 0, "eps_abs": 1e-8, "eps_rel": 1e-8}
            | changes
        )
    )
    return settings


def _solve(data, settings, workspace):
    return solve_torch_osqp_direct(
        *(torch.as_tensor(array, dtype=torch.float64) for array in data),
        settings,
        workspace,
    )


def _bits(actual, expected):
    left, right = (np.asarray(v, dtype="<f8") for v in (actual, expected))
    assert left.shape == right.shape
    assert left.tobytes() == right.tobytes()


def _failure(error, workspace, native_step, stage):
    assert error.stage == stage
    assert error.exit_code == native_step["rc"]
    info = error.info
    assert workspace.last_info == info
    assert not info["status_compatible"]
    assert (
        info["status_val"],
        info["admm_iterations"],
        info["status_polish"],
        info["rho_updates"],
    ) == (
        native_step["status_val"],
        native_step["iter"],
        native_step["status_polish"],
        native_step["rho_updates"],
    )
    assert workspace.rho_updates_committed == native_step["rho_updates"]
    assert workspace.last_rel_kkt == native_step["history"]
    for torch_key, c_key in (
        ("objective", "obj_val"),
        ("dual_objective", "dual_obj_val"),
        ("primal_residual", "prim_res"),
        ("dual_residual", "dual_res"),
        ("duality_gap", "duality_gap"),
        ("rel_kkt_error", "rel_kkt_error"),
        ("rho_bar", "rho"),
        ("rho_estimate", "rho_estimate"),
        ("public_x", "public_x"),
        ("public_y", "public_y"),
    ):
        _bits(info[torch_key], native_step[c_key])
    for key in ("x", "z", "y"):
        _bits(workspace.work_iterates[key], native_step[key])


def _retry(solution, info, expected):
    assert info["status_val"] == expected["status_val"] == 1
    assert info["admm_iterations"] == expected["iter"]
    assert info["status_polish"] == expected["status_polish"]
    assert info["rho_updates"] == expected["rho_updates"]
    _bits(solution.reshape(-1), expected["public_x"])
    _bits(info["public_y"], expected["public_y"])


@needs_dll
@pytest.mark.parametrize(("scaling", "bad"), [(0, -2.0), (0, -0.5), (1, -2.0)])
def test_failing_matrix_callback_keeps_attempted_data_and_repairs(scaling, bad):
    c = matrix(scaling, bad)
    assert c["failed"]["rc"] == 1 and c["failed"]["status_val"] == 11
    original = ([[2.0]], [-1.0], [[1.0]], [-1.0], [1.0])
    ws = TorchOSQPWorkspace()
    cfg = _settings(scaling=scaling, sigma=0.5)
    _retry(*_solve(original, cfg, ws), c["initial"])
    with pytest.raises(TorchLinearSolveError) as failure:
        _solve(([[bad]], *original[1:]), cfg, ws)
    _failure(failure.value, ws, c["failed"], "matrix_update")
    assert ws.status_problem[0].to_dense().item() == bad
    assert not ws.linear_solver.ready
    solution, info = _solve(original, cfg, ws)
    assert info["factorizations_this_solve"] == 1
    _retry(solution, info, c["retry"])


@needs_dll
def test_explicit_rho_failure_preserves_status_and_rebuilds_on_repair():
    c = rho(False)
    assert c["failed"]["rc"] == 1 and c["failed"]["status_val"] == 1
    data = ([[-1.5]], [0.0], [[1.0]], [-1.0], [1.0])
    ws = TorchOSQPWorkspace()
    cfg = _settings(sigma=0.5, rho=2.0)
    _retry(*_solve(data, cfg, ws), c["initial"])
    with pytest.raises(TorchLinearSolveError) as failure:
        _solve(data, cfg | {"rho": 1.0}, ws)
    _failure(failure.value, ws, c["failed"], "rho_update")
    assert ws.rho_bar == 1.0 and not ws.linear_solver.ready
    solution, info = _solve(data, cfg, ws)
    assert info["factorizations_this_solve"] == 1
    _retry(solution, info, c["retry"])


@needs_dll
def test_bound_callback_failure_commits_new_q_and_bounds_before_repair():
    c = rho(True)
    assert c["failed"]["rc"] == 1 and c["failed"]["status_val"] == 11
    first = ([[-1.5]], [0.0], [[1.0]], [0.0], [0.0])
    failed = (first[0], [0.25], first[2], first[3], [1.0])
    repair = (*failed[:4], [0.0])
    cfg = _settings(sigma=0.5, rho=1.0)
    ws = TorchOSQPWorkspace()
    _retry(*_solve(first, cfg, ws), c["initial"])
    with pytest.raises(TorchLinearSolveError) as error:
        _solve(failed, cfg, ws)
    _failure(error.value, ws, c["failed"], "bounds_update")
    assert ws.status_problem[1].item() == 0.25 and ws.status_problem[4].item() == 1.0
    assert ws.constraint_types.tolist() == [0]
    assert not ws.linear_solver.ready
    _retry(*_solve(repair, cfg, ws), c["retry"])


@needs_dll
def test_polishing_init_failure_keeps_admm_state_and_previous_publication():
    c = polish()
    assert c["failed"]["rc"] == 4 and c["failed"]["status_polish"] == -2
    first = ([[-2.0]], [-0.25], [[1.0]], [0.0], [0.0])
    updated = (first[0], [0.0], *first[2:])
    cfg = _settings(sigma=1.0)
    ws = TorchOSQPWorkspace()
    _retry(*_solve(first, cfg, ws), c["initial"])
    factors = ws.linear_solver.factors
    with pytest.raises(TorchLinearSolveError) as failure:
        _solve(updated, cfg | {"polishing": True}, ws)
    _failure(failure.value, ws, c["failed"], "polishing")
    assert ws.linear_solver.factors is factors
    _retry(*_solve(updated, cfg, ws), c["retry"])


@needs_dll
def test_first_polishing_failure_keeps_setup_public_solution_buffers():
    problem = ([[-2.0]], [0.0], [[1.0]], [0.0], [0.0])
    with native(problem, sigma=1.0, polishing=True) as session:
        c = step(session, session.solve)
    assert (c["rc"], c["status_val"], c["status_polish"]) == (4, 1, -2)
    _bits(c["public_x"], [0.])
    _bits(c["public_y"], [0.])

    ws = TorchOSQPWorkspace()
    with pytest.raises(TorchLinearSolveError) as failure:
        _solve(problem, _settings(sigma=1.0, polishing=True), ws)
    _failure(failure.value, ws, c, "polishing")


@needs_dll
def test_adaptive_callback_error_keeps_failed_attempt_count_and_last_successful_history(
    monkeypatch,
):
    c = adapt()
    assert c["failed"]["rc"] == 1 and c["failed"]["rho_updated"] == 1
    first = (
        [[4.0, 1.0], [1.0, 2.0]],
        [1.0, 1.0],
        [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        [1.0, 0.0, 0.0, -1e30],
        [1.0, 0.7, 0.7, 1e30],
    )
    updated = (first[0], [0.0, 1.0], *first[2:])
    cfg = _settings(rho=1e-4, adaptive_rho=1, adaptive_rho_interval=25, max_iter=2000)
    ws = TorchOSQPWorkspace()
    _retry(*_solve(first, cfg, ws), c["initial"])
    import pygranso.private.torchQDLDLSolve as linsys

    def fail(_solver, _matrix, **kwargs):
        raise linsys._QDLDLNumericError("controlled rho factorization failure")

    with monkeypatch.context() as patch:
        patch.setattr(QDLDLSolver, "factorize", fail)
        with pytest.raises(TorchLinearSolveError) as failure:
            _solve(updated, cfg, ws)
    _failure(failure.value, ws, c["failed"], "adaptive_rho")
    assert ws.rho_bar == c["callback"][0]["rho"] and not ws.linear_solver.ready
    _retry(*_solve(updated, cfg, ws), c["retry"])


@needs_dll
def test_explicit_adapter_restores_last_good_matrix_and_forwards_callback_error():
    ws = TorchOSQPWorkspace()
    device = torch.device("cpu")
    cfg = _settings(sigma=0.5)
    options = {"algebra": "torch", "workspace": ws, "settings": cfg}
    P = torch.tensor([[2.0]], dtype=torch.float64)
    q = P.new_tensor([[-1.0]])
    l, u = q.new_tensor([[-1.0]]), q.new_tensor([[1.0]])

    def call(p):
        return solveQP(p, q, None, None, l, u, "osqp", device, True, options)

    initial, _ = call(P)
    state, factors, matrix = ws.state, ws.linear_solver.factors, ws.linear_solver.matrix
    count = ws.linear_solver.factorization_count
    c = matrix_reference()
    with pytest.raises(TorchLinearSolveError, match="KKT inertia") as failure:
        call(-P)
    info = getLastOSQPInfo(ws)
    assert info["failure_stage"] == failure.value.stage == "matrix_update"
    assert info["native_exit_code"] == c["rc"] and info["status_val"] == c["status_val"]
    assert ws.state is state and ws.linear_solver.factors is factors
    assert (
        ws.linear_solver.matrix is matrix
        and ws.linear_solver.factorization_count == count
    )
    recovered, retry = call(P)
    assert retry["factorization_reused"] and retry["status_val"] == 1
    torch.testing.assert_close(recovered, initial, rtol=0, atol=2e-16)


def matrix_reference():
    return matrix(0, -2.0)["failed"]


@needs_dll
@pytest.mark.parametrize("stage", ["rho_update", "polishing"])
def test_explicit_adapter_callback_diagnostic_and_warm_recovery(stage):
    ws = TorchOSQPWorkspace()
    if stage == "rho_update":
        c = rho(False)
        recovered = c["retry"]
        p, q, l, u = -1.5, 0.0, -1.0, 1.0
        original, attempted = _settings(sigma=0.5, rho=2.0), {"rho": 1.0}
    else:
        c = polish()
        p, q, l, u = -2.0, -0.25, 0.0, 0.0
        original, attempted = _settings(sigma=1.0), {"polishing": True}
        # Compare adapter recovery with a native run from the original solved state.
        with native(([[p]], [q], [[1.0]], [l], [u]), sigma=1.0) as retained:
            assert retained.solve()["solve_code"] == 0
            assert retained.update(q=[0.0]) == 0
            recovered = step(retained, retained.solve)
    p = torch.tensor([[p]], dtype=torch.float64)
    q, l, u = (p.new_tensor([[value]]) for value in (q, l, u))

    def call(linear, settings):
        options = {"algebra": "torch", "workspace": ws, "settings": settings}
        return solveQP(p, linear, None, None, l, u, "osqp", p.device, True, options)

    _retry(*call(q, original), c["initial"])
    state, committed, factors = ws.state, ws.solver_info, ws.linear_solver.factors
    requests, failures = getErr()
    linear = torch.zeros_like(q)
    with pytest.raises(TorchLinearSolveError) as error:
        call(linear, original | attempted)
    assert getErr() == [requests + 1, failures + 1]
    assert ws.state is state and ws.solver_info is committed
    assert ws.linear_solver.factors is factors
    info = getLastOSQPInfo(ws)
    assert info["failure_stage"] == error.value.stage == stage
    expected = c["failed"]
    assert (info["native_exit_code"], info["status_val"], info["status_polish"]) == (
        expected["rc"],
        expected["status_val"],
        expected["status_polish"],
    )
    _bits(info["public_x"], expected["public_x"])
    _bits(info["public_y"], expected["public_y"])
    solution, retry = call(linear, original)
    assert retry["status_val"] == 1 and retry["factorization_reused"]
    assert retry["status_polish"] == committed["status_polish"]
    _retry(solution, retry, recovered)


def test_scaled_adapter_failure_does_not_mutate_last_good_bound_storage():
    ws = TorchOSQPWorkspace()
    p = torch.tensor([[-1.0]], dtype=torch.float64)
    q, l, u = p.new_zeros((1, 1)), p.new_zeros((1, 1)), p.new_zeros((1, 1))
    options = {
        "algebra": "torch",
        "workspace": ws,
        "settings": _settings(scaling=1, sigma=0.5, rho=0.5),
    }

    def solve(upper):
        return solveQP(p, q, None, None, l, upper, "osqp", p.device, True, options)

    _, info = solve(u)
    assert info["status_val"] == 1
    scaling, public, factor = ws.scaling, ws.state, ws.linear_solver.factors
    before = scaling["scaled_u"]
    with pytest.raises(TorchLinearSolveError):
        solve(p.new_ones((1, 1)))
    assert ws.scaling is scaling and scaling["scaled_u"] is before
    assert torch.equal(scaling["scaled_u"], u.reshape(-1))
    assert ws.state is public and ws.linear_solver.factors is factor
    _, retry = solve(u)
    assert retry["status_val"] == 1 and retry["factorization_reused"]


def test_rejected_adapter_candidate_does_not_replace_committed_polish_status():
    ws = TorchOSQPWorkspace()

    def solve(p, q, **changes):
        options = {
            "algebra": "torch",
            "workspace": ws,
            "settings": _settings(**changes),
        }
        l, u = -torch.ones_like(q), torch.ones_like(q)
        return solveQP(p, q, None, None, l, u, "osqp", p.device, True, options)

    p = torch.eye(1, dtype=torch.float64)
    q = p.new_zeros((1, 1))
    _, first = solve(p, q, polishing=1)
    assert first["status_val"] == 1 and first["status_polish"] == 2
    with pytest.raises(RuntimeError, match="max_iter_reached"):
        solve(torch.eye(2, dtype=torch.float64), p.new_full((2, 1), 2.0), max_iter=1)
    assert getLastOSQPInfo(ws)["status_polish"] == 0
    _, retry = solve(p, q)
    assert retry["factorization_reused"] and retry["status_polish"] == 2


@pytest.mark.parametrize("algebra", ["builtin", "auto"])
def test_optimizer_raises_on_incompatible_builtin_outcome(monkeypatch, algebra):
    from pygranso.private import osqpTorchAdapter as adapter

    def failed_backend(p, *args, **kwargs):
        return p.new_full((1, 1), 0.4), {
            "status": "max_iter_reached",
            "status_val": 7,
            "status_compatible": False,
        }

    monkeypatch.setattr(adapter, "_solve_builtin_osqp_path", failed_backend)
    ws = TorchOSQPWorkspace()
    p = torch.eye(1, dtype=torch.float64)
    q, l, u = p.new_zeros((1, 1)), -p, p
    options = {"algebra": algebra, "workspace": ws}
    requests, failures = getErr()
    with pytest.raises(RuntimeError, match="max_iter_reached"):
        solveQP(p, q, None, None, l, u, "osqp", p.device, True, options)
    assert getErr() == [requests + 1, failures + 1]
    assert getLastOSQPInfo(ws)["status_val"] == 7
