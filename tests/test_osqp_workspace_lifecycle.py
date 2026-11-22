import json
import math
from fractions import Fraction

import pytest
import torch

import pygranso.private.torchOSQP as torch_osqp_module
from pygranso.private.osqpTorchAdapter import DEFAULT_OSQP_SETTINGS, solve_osqp_torch_qp
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchLinearSolve import TorchLinearSolveError
from pygranso.private.torchOSQP import _prepare_workspace, solve_torch_osqp_direct


def _seed_workspace(workspace):
    workspace.state = {
        "x": torch.ones(2),
        "z": torch.ones(2),
        "y": torch.ones(2),
    }
    workspace.rho_bar = 0.25
    workspace.rho_setting = 0.1


def test_workspace_reset_clears_adaptive_rho_scheduler_state():
    workspace = TorchOSQPWorkspace()
    workspace.rho_setup_signature = (3, 1, 0.4, 5.0)
    workspace.adaptive_rho_interval = 1
    workspace.adaptive_rho_setup_time = 0.25
    workspace.last_rel_kkt = 2.0

    workspace.reset_torch()

    assert workspace.rho_setup_signature is None
    assert workspace.adaptive_rho_interval is None
    assert workspace.adaptive_rho_setup_time is None
    assert workspace.last_rel_kkt is None


@pytest.mark.parametrize("changed_matrix", ["P", "A"])
def test_nonconvex_matrix_update_restores_workspace_and_factor_reuse(
    changed_matrix, floating_dtype, available_device, monkeypatch
):
    # Unit-test the C factor/update rule beneath the stricter input gate.
    # Public validation/rollback is covered by the solveQP failure-contract test.
    monkeypatch.setattr(torch_osqp_module, "_validate_qp", lambda *args: args[:5])
    P = torch.tensor(
        [[1.0 if changed_matrix == "P" else -1.0]],
        dtype=floating_dtype,
        device=available_device,
    )
    A = P.new_full((1, 1), 1.0 if changed_matrix == "P" else 10.0)
    q, l, u = P.new_zeros(1), -P.new_ones(1), P.new_ones(1)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": 0,
        "adaptive_rho": False,
        "polishing": False,
        "check_termination": 1,
        "return_info": True,
    }
    workspace = TorchOSQPWorkspace()
    _, initial = solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)
    assert initial["status"] == "solved"
    fields_before = workspace.__dict__.copy()
    solver_fields_before = workspace.linear_solver.__dict__.copy()
    P_next = -P if changed_matrix == "P" else P
    A_next = A / 10.0 if changed_matrix == "A" else A
    with pytest.raises(TorchLinearSolveError, match="KKT inertia"):
        solve_torch_osqp_direct(P_next, q, A_next, l, u, settings, workspace)

    for name, value in fields_before.items():
        if name == "builtin_stats":
            assert workspace.builtin_stats == value
        else:
            assert workspace.__dict__[name] is value
    for name, value in solver_fields_before.items():
        assert workspace.linear_solver.__dict__[name] is value

    def no_recheck(*args, **kwargs):
        pytest.fail("An unchanged, previously validated KKT must reuse its check")

    monkeypatch.setattr(torch.linalg, "eigvalsh", no_recheck)
    _, reused = solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)
    assert reused["status"] == "solved"
    assert reused["factorization_reused"]
    assert reused["factorizations_this_solve"] == 0


def test_rho_only_refactor_preserves_pinned_distinct_inertia_policy(monkeypatch):
    # Deliberately inject nonconvex data past validation to isolate the C rule.
    monkeypatch.setattr(torch_osqp_module, "_validate_qp", lambda *args: args[:5])
    P = torch.tensor([[-1.0]], dtype=torch.float64)
    A, q = P.new_full((1, 1), 10.0), P.new_zeros(1)
    l, u = -P.new_ones(1), P.new_ones(1)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": 0,
        "adaptive_rho": False,
        "polishing": False,
        "check_termination": 1,
        "return_info": True,
    }
    workspace = TorchOSQPWorkspace()
    solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)

    def no_matrix_update_check(*args, **kwargs):
        pytest.fail("Pinned rho-only updates check factor failure, not positive count")

    monkeypatch.setattr(torch.linalg, "eigvalsh", no_matrix_update_check)
    settings["rho"] = 0.001
    _, info = solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)
    assert info["status"] == "solved"
    assert info["factorizations_this_solve"] == 1
    assert not info["factorization_reused"]
    assert info["rho_bar"] == 0.001


@pytest.mark.parametrize("rho_mode", [2, 3])
def test_adaptive_rho_refactor_failure_restores_previous_workspace(
    rho_mode, floating_dtype, available_device, monkeypatch
):
    P = torch.ones((1, 1), dtype=floating_dtype, device=available_device)
    A = P.clone()
    q = P.new_zeros(1)
    l, u = -P.new_ones(1), P.new_ones(1)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": 10,
        "adaptive_rho": False,
        "polishing": False,
        "check_termination": 1,
        "return_info": True,
    }
    workspace = TorchOSQPWorkspace()
    _, initial = solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)
    assert initial["status"] == "solved"

    fields_before = workspace.__dict__.copy()
    solver_fields_before = workspace.linear_solver.__dict__.copy()

    def fail_refactorization(matrix, **kwargs):
        raise TorchLinearSolveError("synthetic adaptive-rho refactorization failure")

    monkeypatch.setattr(workspace.linear_solver, "factorize", fail_refactorization)
    settings.update(
        {
            "adaptive_rho": rho_mode,
            "rho_update_interval": 1,
            "check_termination": 0,
            "max_iter": 1,
        }
    )
    with pytest.raises(TorchLinearSolveError, match="adaptive-rho refactorization"):
        solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)

    for name, value in fields_before.items():
        if name == "builtin_stats":
            assert workspace.builtin_stats == value
        else:
            assert workspace.__dict__[name] is value
    for name, value in solver_fields_before.items():
        assert workspace.linear_solver.__dict__[name] is value


@pytest.mark.parametrize(
    "changed_key,changed_value",
    [
        ("adaptive_rho", 3),
        ("rho_update_interval", 7),
        ("adaptive_rho_fraction", 0.8),
        ("rho_update_tolerance", 7.0),
    ],
)
def test_setup_only_rho_setting_change_forces_fresh_torch_setup(
    changed_key, changed_value, monkeypatch,
):
    P = torch.eye(1, dtype=torch.float64)
    q = torch.tensor([-1.0], dtype=torch.float64)
    A = torch.eye(1, dtype=torch.float64)
    l, u = torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "adaptive_rho": 1,
        "polishing": False,
        "return_info": True,
    }
    workspace = TorchOSQPWorkspace()
    solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)
    old_state = workspace.state
    observed = {}
    original = torch_osqp_module._initial_scaled_state

    def capture(active_workspace, *args, **kwargs):
        observed["fresh"] = active_workspace.state is None
        return original(active_workspace, *args, **kwargs)

    monkeypatch.setattr(torch_osqp_module, "_initial_scaled_state", capture)
    settings[changed_key] = changed_value
    solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)

    assert old_state is not None
    assert observed == {"fresh": True}
    assert workspace.state is not old_state


@pytest.mark.parametrize(
    "update_p,update_a",
    [(True, False), (False, True), (True, True)],
    ids=["P-values", "A-values", "P-and-A-values"],
)
def test_compatible_matrix_value_update_preserves_warm_state(update_p, update_a):
    workspace = TorchOSQPWorkspace()
    P = torch.eye(2, dtype=torch.float64)
    A = torch.eye(2, dtype=torch.float64)
    _prepare_workspace(workspace, P, A, ("bounds", 2))
    _seed_workspace(workspace)
    state_before = {key: value.clone() for key, value in workspace.state.items()}
    rho_before = workspace.rho_bar
    workspace.scaling = {"sentinel": True}
    workspace.scaling_source_p = P.clone()
    workspace.scaling_source_a = A.clone()
    workspace.scaling_passes = 1

    updated_p = 2.0 * P if update_p else P
    updated_a = 3.0 * A if update_a else A
    _prepare_workspace(workspace, updated_p, updated_a, ("bounds", 2))

    for key, value in state_before.items():
        assert torch.equal(workspace.state[key], value)
    assert workspace.rho_bar == rho_before
    assert workspace.scaling is None
    assert workspace.scaling_source_p is None
    assert workspace.scaling_source_a is None
    assert workspace.scaling_passes == 0


def test_scaled_matrix_value_update_refactorizes_when_scaled_kkt_is_unchanged(
    floating_dtype, available_device, monkeypatch
):
    P = torch.zeros((1, 1), dtype=floating_dtype, device=available_device)
    q = P.new_tensor([-1.0])
    A = P.new_ones((1, 1))
    l = P.new_zeros(1)
    u = P.new_ones(1)
    tolerance = 5e-4 if floating_dtype == torch.float32 else 5e-8
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": 1,
        "adaptive_rho": False,
        "polishing": False,
        "check_termination": 1,
        "eps_abs": tolerance,
        "eps_rel": tolerance,
        "return_info": True,
    }
    workspace = TorchOSQPWorkspace()

    first_solution, first_info = solve_torch_osqp_direct(
        P, q, A, l, u, settings, workspace
    )
    # The pinned float build reaches the cap at x=1.0625 on the first LP.
    if floating_dtype == torch.float32:
        assert first_info["status_val"] == 7
        assert first_info["admm_iterations"] == settings["max_iter"]
        assert first_solution.item() == 1.0625
    else:
        assert first_info["status"] == "solved"
        assert torch.allclose(first_solution, P.new_tensor([1.0]), atol=tolerance, rtol=tolerance)
    state_before = workspace.state
    scaling_before = workspace.scaling
    kkt_before = workspace.linear_solver.matrix.clone()
    observed = {}
    original = torch_osqp_module._initial_scaled_state

    def capture_warm_state(active_workspace, *args, **kwargs):
        observed["preserved"] = active_workspace.state is state_before
        return original(active_workspace, *args, **kwargs)

    monkeypatch.setattr(
        torch_osqp_module, "_initial_scaled_state", capture_warm_state
    )
    updated_solution, updated_info = solve_torch_osqp_direct(
        P, q, 4.0 * A, l, u, settings, workspace
    )

    assert updated_info["status"] == "solved"
    assert torch.allclose(
        updated_solution, P.new_tensor([0.25]), atol=tolerance, rtol=tolerance
    )
    assert observed == {"preserved": True}
    assert workspace.scaling is not scaling_before
    assert not torch.equal(workspace.scaling["E"], scaling_before["E"])
    assert torch.equal(workspace.linear_solver.matrix, kkt_before)
    assert not updated_info["factorization_reused"]
    assert updated_info["factorizations_this_solve"] == 1


def test_scaled_matrix_value_refactor_failure_restores_previous_workspace(monkeypatch):
    dtype = torch.float64
    P = torch.zeros((1, 1), dtype=dtype)
    q = torch.tensor([-1.0], dtype=dtype)
    A = torch.ones((1, 1), dtype=dtype)
    l = torch.zeros(1, dtype=dtype)
    u = torch.ones(1, dtype=dtype)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": 1,
        "adaptive_rho": False,
        "polishing": False,
        "check_termination": 1,
        "return_info": True,
    }
    workspace = TorchOSQPWorkspace()
    solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)
    fields_before = workspace.__dict__.copy()
    solver_fields_before = workspace.linear_solver.__dict__.copy()

    def fail_factorization(matrix, **kwargs):
        raise TorchLinearSolveError("synthetic scaled matrix-update failure")

    monkeypatch.setattr(workspace.linear_solver, "factorize", fail_factorization)
    with pytest.raises(TorchLinearSolveError, match="scaled matrix-update"):
        solve_torch_osqp_direct(P, q, 4.0 * A, l, u, settings, workspace)

    for name, value in fields_before.items():
        if name == "builtin_stats":
            assert workspace.builtin_stats == value
        else:
            assert workspace.__dict__[name] is value
    for name, value in solver_fields_before.items():
        assert workspace.linear_solver.__dict__[name] is value


@pytest.mark.parametrize(
    "failure_stage,exception_type,message",
    [
        ("scaling", MemoryError, "synthetic scaling allocation failure"),
        (
            "scaling",
            torch.OutOfMemoryError,
            "synthetic device allocation failure",
        ),
        ("solve", TorchLinearSolveError, "synthetic main solve failure"),
        ("finalize", RuntimeError, "synthetic final diagnostics failure"),
    ],
)
def test_failed_matrix_update_restores_committed_workspace_at_each_solve_stage(
    failure_stage, exception_type, message, monkeypatch
):
    dtype = torch.float64
    P = torch.ones((1, 1), dtype=dtype)
    q = torch.tensor([-0.5], dtype=dtype)
    A = torch.ones((1, 1), dtype=dtype)
    l = torch.zeros(1, dtype=dtype)
    u = torch.ones(1, dtype=dtype)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": 1,
        "adaptive_rho": False,
        "polishing": False,
        "check_termination": 1,
        "return_info": True,
    }
    workspace = TorchOSQPWorkspace()
    _, initial = solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)
    assert initial["status"] == "solved"
    fields_before = workspace.__dict__.copy()
    solver_fields_before = workspace.linear_solver.__dict__.copy()

    def fail(*args, **kwargs):
        raise exception_type(message)

    with monkeypatch.context() as failure:
        if failure_stage == "scaling":
            failure.setattr(torch_osqp_module, "_scale_problem", fail)
        elif failure_stage == "solve":
            failure.setattr(workspace.linear_solver, "solve", fail)
        else:
            failure.setattr(torch_osqp_module, "_compute_rho_estimate", fail)
        with pytest.raises(exception_type, match=message):
            solve_torch_osqp_direct(P, q, 2.0 * A, l, u, settings, workspace)

    for name, value in fields_before.items():
        if name == "builtin_stats":
            assert workspace.builtin_stats == value
        else:
            assert workspace.__dict__[name] is value
    for name, value in solver_fields_before.items():
        assert workspace.linear_solver.__dict__[name] is value

    _, reused = solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)
    assert reused["status"] == "solved"
    assert reused["factorization_reused"]
    assert reused["factorizations_this_solve"] == 0


def test_bound_pair_permutation_on_duplicate_rows_reuses_warm_kkt(
    floating_dtype, available_device, monkeypatch
):
    """Pinned l/u updates do not become matrix updates when row types stay fixed."""
    P = torch.ones((1, 1), device=available_device, dtype=floating_dtype)
    q = P.new_tensor([-0.5])
    A = P.new_ones((2, 1))
    l = P.new_tensor([-1.0, 0.0])
    u = P.new_tensor([1.0, 2.0])
    solver_tolerance = 1e-5 if floating_dtype == torch.float32 else 1e-8
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": 0,
        "adaptive_rho": False,
        "polishing": False,
        "check_termination": 1,
        "eps_abs": solver_tolerance,
        "eps_rel": solver_tolerance,
        "return_info": True,
    }
    workspace = TorchOSQPWorkspace()

    first_solution, first_info = solve_torch_osqp_direct(
        P, q, A, l, u, settings, workspace
    )
    assert first_info["status"] == "solved"
    state_before = workspace.state
    observed = {}
    original = torch_osqp_module._initial_scaled_state

    def capture_warm_state(active_workspace, *args, **kwargs):
        observed["preserved"] = active_workspace.state is state_before
        return original(active_workspace, *args, **kwargs)

    monkeypatch.setattr(
        torch_osqp_module, "_initial_scaled_state", capture_warm_state
    )
    second_solution, second_info = solve_torch_osqp_direct(
        P, q, A, l.flip(0), u.flip(0), settings, workspace
    )

    assert second_info["status"] == "solved"
    solution_tolerance = 5e-4 if floating_dtype == torch.float32 else 5e-8
    assert torch.allclose(
        second_solution,
        first_solution,
        atol=solution_tolerance,
        rtol=solution_tolerance,
    )
    assert observed == {"preserved": True}
    assert second_info["factorization_reused"]
    assert second_info["factorizations_this_solve"] == 0


def test_structure_change_invalidates_complete_torch_state():
    workspace = TorchOSQPWorkspace()
    P = torch.eye(2, dtype=torch.float64)
    A = torch.eye(2, dtype=torch.float64)
    _prepare_workspace(workspace, P, A, ("bounds", 2))
    _seed_workspace(workspace)
    changed = A.clone()
    changed[0, 1] = 1.0
    _prepare_workspace(workspace, P, changed, ("bounds", 2))
    assert workspace.state is None
    assert workspace.rho_bar is None
    assert not workspace.linear_solver.ready


def test_same_nnz_pattern_change_resets_state_and_matches_fresh_setup(monkeypatch):
    dtype = torch.float64
    P = torch.eye(2, dtype=dtype)
    q = -torch.ones(2, dtype=dtype)
    A = torch.eye(2, dtype=dtype)
    changed_a = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=dtype)
    assert torch.count_nonzero(A) == torch.count_nonzero(changed_a)
    l = torch.tensor([0.0, -torch.inf], dtype=dtype)
    u = torch.tensor([1.0, torch.inf], dtype=dtype)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": 0,
        "adaptive_rho": False,
        "polishing": False,
        "check_termination": 1,
        "return_info": True,
    }
    workspace = TorchOSQPWorkspace()
    solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)
    observed = {}
    original = torch_osqp_module._initial_scaled_state

    def capture_cold_state(active_workspace, *args, **kwargs):
        if active_workspace is workspace:
            observed.update(
                state=active_workspace.state,
                rho_bar=active_workspace.rho_bar,
                solver_ready=active_workspace.linear_solver.ready,
            )
        return original(active_workspace, *args, **kwargs)

    monkeypatch.setattr(
        torch_osqp_module, "_initial_scaled_state", capture_cold_state
    )
    updated_solution, updated_info = solve_torch_osqp_direct(
        P, q, changed_a, l, u, settings, workspace
    )
    fresh_solution, fresh_info = solve_torch_osqp_direct(
        P, q, changed_a, l, u, settings, TorchOSQPWorkspace()
    )

    assert observed == {"state": None, "rho_bar": None, "solver_ready": False}
    assert updated_info["status"] == fresh_info["status"] == "solved"
    assert updated_info["factorizations_this_solve"] == 1
    assert torch.allclose(updated_solution, fresh_solution, atol=1e-8, rtol=1e-8)
    assert updated_info["objective"] == pytest.approx(
        fresh_info["objective"], abs=1e-8
    )


def test_constraint_order_signature_change_invalidates_complete_torch_state():
    workspace = TorchOSQPWorkspace()
    P = torch.eye(2, dtype=torch.float64)
    A = torch.ones((2, 2), dtype=torch.float64)
    _prepare_workspace(workspace, P, A, ("c1", "c2"))
    _seed_workspace(workspace)
    _prepare_workspace(workspace, P, A, ("c2", "c1"))
    assert workspace.state is None
    assert workspace.constraint_order_signature == ("c2", "c1")


def test_dense_row_permutation_invalidates_warm_state_without_pattern_change():
    workspace = TorchOSQPWorkspace()
    P = torch.eye(2, dtype=torch.float64)
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    signature = ("equalities", 2)
    _prepare_workspace(workspace, P, A, signature)
    _seed_workspace(workspace)
    _prepare_workspace(workspace, P, A.flip(0), signature)
    assert workspace.state is None
    assert workspace.rho_bar is None


def test_equality_or_loose_classification_change_preserves_warm_state_and_rho():
    workspace = TorchOSQPWorkspace()
    P = torch.eye(2, dtype=torch.float64)
    A = torch.eye(2, dtype=torch.float64)
    l = torch.tensor([-torch.inf, 0.0], dtype=torch.float64)
    u = torch.tensor([torch.inf, 1.0], dtype=torch.float64)
    _prepare_workspace(workspace, P, A, ("bounds", 2), l, u)
    _seed_workspace(workspace)
    matrix = torch.cat((torch.cat((P, A.T), 1), torch.cat((A, -P), 1)))
    workspace.linear_solver.factorize(matrix, P=P, A=A, sigma=0.0, rho_inv_vec=P.diag(), p_pattern=P.ne(0), a_pattern=A.ne(0))
    state_before = workspace.state
    rho_bar_before = workspace.rho_bar
    rho_setting_before = workspace.rho_setting
    changed_l = torch.tensor([0.0, 0.0], dtype=torch.float64)
    changed_u = torch.tensor([0.0, 1.0], dtype=torch.float64)
    _prepare_workspace(
        workspace,
        P,
        A,
        ("bounds", 2),
        changed_l,
        changed_u,
    )
    assert workspace.state is state_before
    assert workspace.rho_bar == rho_bar_before
    assert workspace.rho_setting == rho_setting_before
    assert not workspace.linear_solver.ready
    assert torch.equal(workspace.constraint_types, torch.tensor([1, 0]))


def test_scaled_constraint_type_change_preserves_warm_state_and_rho(
    floating_dtype, available_device, monkeypatch
):
    dtype = floating_dtype
    device = available_device
    workspace = TorchOSQPWorkspace()
    P = torch.ones((1, 1), dtype=dtype, device=device)
    q = torch.zeros(1, dtype=dtype, device=device)
    A = torch.full((1, 1), 1e8, dtype=dtype, device=device)
    l = torch.zeros(1, dtype=dtype, device=device)
    settings = {
        "rho": 0.1,
        "sigma": 1e-6,
        "alpha": 1.6,
        "max_iter": 100,
        "eps_abs": 1e-6,
        "eps_rel": 1e-6,
        "eps_prim_inf": 1e-4,
        "eps_dual_inf": 1e-4,
        "check_termination": 1,
        "scaling": 1,
        "adaptive_rho": False,
        "rho_update_interval": 50,
        "rho_update_tolerance": 5.0,
        "warm_start": True,
        "polishing": False,
        "return_info": True,
    }
    _, first = solve_torch_osqp_direct(
        P,
        q,
        A,
        l,
        torch.tensor([2e-4], dtype=dtype, device=device),
        settings,
        workspace,
    )
    assert first["status_compatible"]
    assert workspace.constraint_types.item() == 1
    assert first["rho_min"] == pytest.approx(1000.0 * first["rho_bar"])
    assert first["rho_max"] == pytest.approx(1000.0 * first["rho_bar"])
    state_before = workspace.state
    rho_bar_before = workspace.rho_bar
    rho_setting_before = workspace.rho_setting

    observed = {}
    original = torch_osqp_module._initial_scaled_state

    def capture_initial_state(active_workspace, *args, **kwargs):
        observed["state_preserved"] = active_workspace.state is state_before
        observed["rho_bar"] = active_workspace.rho_bar
        observed["rho_setting"] = active_workspace.rho_setting
        observed["solver_ready"] = active_workspace.linear_solver.ready
        return original(active_workspace, *args, **kwargs)

    monkeypatch.setattr(
        torch_osqp_module, "_initial_scaled_state", capture_initial_state
    )
    _, second = solve_torch_osqp_direct(
        P,
        q,
        A,
        l,
        torch.tensor([2.0], dtype=dtype, device=device),
        settings,
        workspace,
    )

    assert second["status_compatible"]
    assert workspace.constraint_types.item() == 0
    assert second["rho_min"] == pytest.approx(second["rho_bar"])
    assert second["rho_max"] == pytest.approx(second["rho_bar"])
    assert observed == {
        "state_preserved": True,
        "rho_bar": rho_bar_before,
        "rho_setting": rho_setting_before,
        "solver_ready": False,
    }
    assert not second["factorization_reused"]
    assert second["factorizations_this_solve"] == 1


def test_type_change_builds_row_rho_from_preserved_adapted_rho(
    floating_dtype, available_device, monkeypatch
):
    P = torch.ones((1, 1), dtype=floating_dtype, device=available_device)
    q = P.new_zeros(1)
    A = P.clone()
    inequality_l, inequality_u = -P.new_ones(1), P.new_ones(1)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": 0,
        "max_iter": 1,
        "check_termination": 0,
        "adaptive_rho": True,
        "rho_update_interval": 1,
        "rho_update_tolerance": 5.0,
        "polishing": False,
        "warm_start": True,
        "return_info": True,
    }
    workspace = TorchOSQPWorkspace()

    _, first = solve_torch_osqp_direct(
        P,
        q,
        A,
        inequality_l,
        inequality_u,
        settings,
        workspace,
    )
    assert first["status"] == "solved"
    assert first["rho_updates"] == 1
    assert first["rho_bar"] == pytest.approx(1e-6)
    state_before = workspace.state
    observed = {}
    original = torch_osqp_module._initial_scaled_state

    def capture_initial_state(active_workspace, *args, **kwargs):
        observed["state_preserved"] = active_workspace.state is state_before
        observed["rho_bar"] = active_workspace.rho_bar
        return original(active_workspace, *args, **kwargs)

    monkeypatch.setattr(
        torch_osqp_module, "_initial_scaled_state", capture_initial_state
    )
    settings.update(
        {
            "max_iter": 100,
            "check_termination": 1,
        }
    )
    equality = P.new_zeros(1)
    _, second = solve_torch_osqp_direct(
        P,
        q,
        A,
        equality,
        equality,
        settings,
        workspace,
    )

    assert second["status"] == "solved"
    assert observed == {
        "state_preserved": True,
        "rho_bar": pytest.approx(first["rho_bar"]),
    }
    assert workspace.constraint_types.item() == 1
    assert second["rho_bar"] == pytest.approx(first["rho_bar"])
    assert second["rho_min"] == pytest.approx(1000.0 * first["rho_bar"])
    assert second["rho_max"] == pytest.approx(1000.0 * first["rho_bar"])
    assert not second["factorization_reused"]
    assert second["factorizations_this_solve"] == 1


@pytest.mark.parametrize(
    "initial_type,target_type",
    [
        ("loose", "inequality"),
        ("loose", "equality"),
        ("inequality", "loose"),
        ("inequality", "equality"),
        ("equality", "loose"),
        ("equality", "inequality"),
    ],
)
def test_every_bound_type_transition_preserves_warm_state_and_rebuilds_row_rho(
    initial_type, target_type, monkeypatch
):
    dtype = torch.float64
    P = torch.ones((1, 1), dtype=dtype)
    q = torch.zeros(1, dtype=dtype)
    A = P.clone()
    bounds = {
        "loose": (
            torch.full((1,), -torch.inf, dtype=dtype),
            torch.full((1,), torch.inf, dtype=dtype),
        ),
        "inequality": (-torch.ones(1, dtype=dtype), torch.ones(1, dtype=dtype)),
        "equality": (torch.zeros(1, dtype=dtype), torch.zeros(1, dtype=dtype)),
    }
    expected_type = {"loose": -1, "inequality": 0, "equality": 1}
    expected_rho = {"loose": 1e-6, "inequality": 0.1, "equality": 100.0}
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "rho": 0.1,
        "scaling": 0,
        "adaptive_rho": False,
        "polishing": False,
        "check_termination": 1,
        "warm_start": True,
        "return_info": True,
    }
    workspace = TorchOSQPWorkspace()
    initial_l, initial_u = bounds[initial_type]
    _, first = solve_torch_osqp_direct(
        P, q, A, initial_l, initial_u, settings, workspace
    )
    assert first["status"] == "solved"
    state_before = workspace.state
    observed = {}
    original = torch_osqp_module._initial_scaled_state

    def capture_initial_state(active_workspace, *args, **kwargs):
        observed["state_preserved"] = active_workspace.state is state_before
        observed["solver_ready"] = active_workspace.linear_solver.ready
        return original(active_workspace, *args, **kwargs)

    monkeypatch.setattr(
        torch_osqp_module, "_initial_scaled_state", capture_initial_state
    )
    target_l, target_u = bounds[target_type]
    _, second = solve_torch_osqp_direct(
        P, q, A, target_l, target_u, settings, workspace
    )

    assert second["status"] == "solved"
    assert observed == {"state_preserved": True, "solver_ready": False}
    assert workspace.constraint_types.item() == expected_type[target_type]
    assert second["rho_min"] == pytest.approx(expected_rho[target_type])
    assert second["rho_max"] == pytest.approx(expected_rho[target_type])
    assert not second["factorization_reused"]
    assert second["factorizations_this_solve"] == 1


def test_scaling_configuration_change_invalidates_warm_state():
    workspace = TorchOSQPWorkspace()
    P = torch.eye(2, dtype=torch.float64)
    A = torch.eye(2, dtype=torch.float64)
    _prepare_workspace(
        workspace,
        P,
        A,
        ("bounds", 2),
        scaling_setting=0,
    )
    _seed_workspace(workspace)
    _prepare_workspace(
        workspace,
        P,
        A,
        ("bounds", 2),
        scaling_setting=10,
    )
    assert workspace.state is None
    assert workspace.rho_bar is None


def test_scaled_termination_change_reuses_unchanged_kkt_factors():
    workspace = TorchOSQPWorkspace()
    P = torch.tensor([[1e-12]], dtype=torch.float64)
    q = torch.tensor([-1e-6], dtype=torch.float64)
    A = torch.tensor([[1e-3]], dtype=torch.float64)
    l = torch.tensor([-1e-6], dtype=torch.float64)
    u = -l
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": 10,
        "max_iter": 500,
        "eps_abs": 1e-5,
        "eps_rel": 1e-5,
        "check_termination": 1,
        "adaptive_rho": False,
        "polishing": False,
        "warm_start": False,
        "return_info": True,
        "scaled_termination": False,
    }
    _, unscaled = solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)
    settings["scaled_termination"] = True
    _, scaled = solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)

    assert unscaled["admm_iterations"] == 1
    assert scaled["admm_iterations"] == 63
    assert scaled["factorization_reused"]
    assert scaled["factorizations_this_solve"] == 0
    assert scaled["factorization_count"] == unscaled["factorization_count"] == 1


def test_public_reset_starts_fresh_torch_and_builtin_diagnostics_epoch():
    workspace = TorchOSQPWorkspace()
    matrix = torch.eye(2, dtype=torch.float64)
    A = matrix.new_empty((0, 2))
    workspace.linear_solver.factorize(matrix, P=matrix, A=A, sigma=0.0, rho_inv_vec=A.new_empty(0), p_pattern=matrix.ne(0), a_pattern=A.ne(0))
    workspace.linear_solver.solve(torch.ones(2, dtype=torch.float64))
    workspace.builtin_stats = {
        "setups": 1,
        "updates": 2,
        "rebuilds": 3,
        "last_cache_hit": True,
    }
    workspace.active_backend = "torch"
    workspace.last_info = {"status": "solved"}

    workspace.reset()

    assert not workspace.linear_solver.ready
    assert workspace.linear_solver.factorization_count == 0
    assert workspace.linear_solver.solve_count == 0
    assert workspace.builtin_stats == {
        "setups": 0,
        "updates": 0,
        "rebuilds": 0,
        "last_cache_hit": False,
    }
    assert workspace.active_backend is None
    assert workspace.last_info is None

    workspace.reset()
    assert workspace.linear_solver.factorization_count == 0
    assert workspace.linear_solver.solve_count == 0

    A = matrix.new_empty((0, 2))
    workspace.linear_solver.factorize(matrix, P=matrix, A=A, sigma=0.0, rho_inv_vec=A.new_empty(0), p_pattern=matrix.ne(0), a_pattern=A.ne(0))
    workspace.linear_solver.solve(torch.ones(2, dtype=torch.float64))
    assert workspace.linear_solver.factorization_count == 1
    assert workspace.linear_solver.solve_count == 1


def test_dimension_and_dtype_changes_invalidate_complete_torch_state():
    for changed_p, changed_a in (
        (torch.eye(3, dtype=torch.float64), torch.eye(3, dtype=torch.float64)),
        (torch.eye(2, dtype=torch.float32), torch.eye(2, dtype=torch.float32)),
    ):
        workspace = TorchOSQPWorkspace()
        _prepare_workspace(
            workspace,
            torch.eye(2, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64),
            ("bounds", 2),
        )
        _seed_workspace(workspace)
        _prepare_workspace(workspace, changed_p, changed_a, ("changed",))
        assert workspace.state is None
        assert workspace.rho_bar is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_cuda_transitions_reset_incompatible_state_and_match_fresh_setup(
    monkeypatch,
):
    dtype = torch.float64
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": 0,
        "adaptive_rho": False,
        "polishing": False,
        "check_termination": 1,
        "return_info": True,
    }
    workspace = TorchOSQPWorkspace()
    observed = []
    original = torch_osqp_module._initial_scaled_state

    def capture_state(active_workspace, *args, **kwargs):
        if active_workspace is workspace and observed:
            observed.append(
                (
                    active_workspace.state,
                    active_workspace.rho_bar,
                    active_workspace.linear_solver.ready,
                )
            )
        return original(active_workspace, *args, **kwargs)

    monkeypatch.setattr(torch_osqp_module, "_initial_scaled_state", capture_state)

    def problem(device):
        P = torch.ones((1, 1), dtype=dtype, device=device)
        return (
            P,
            P.new_tensor([-0.5]),
            P.clone(),
            P.new_zeros(1),
            P.new_ones(1),
        )

    solve_torch_osqp_direct(*problem("cpu"), settings, workspace)
    observed.append("begin-transitions")
    cuda_solution, cuda_info = solve_torch_osqp_direct(
        *problem("cuda"), settings, workspace
    )
    cpu_solution, cpu_info = solve_torch_osqp_direct(
        *problem("cpu"), settings, workspace
    )
    fresh_cuda, fresh_cuda_info = solve_torch_osqp_direct(
        *problem("cuda"), settings, TorchOSQPWorkspace()
    )

    assert observed[1:] == [(None, None, False), (None, None, False)]
    assert cuda_info["status"] == fresh_cuda_info["status"] == "solved"
    assert cpu_info["status"] == "solved"
    assert torch.allclose(cuda_solution, fresh_cuda, atol=1e-8, rtol=1e-8)
    assert torch.allclose(cpu_solution, torch.tensor([[0.5]], dtype=dtype), atol=1e-8)
    assert workspace.state["x"].device.type == "cpu"


def test_independent_workspaces_do_not_share_factors_state_or_counters():
    dtype = torch.float64
    P = torch.ones((1, 1), dtype=dtype)
    A = P.clone()
    l, u = P.new_zeros(1), P.new_ones(1)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": 0,
        "adaptive_rho": False,
        "polishing": False,
        "check_termination": 1,
        "return_info": True,
    }
    first_workspace = TorchOSQPWorkspace()
    second_workspace = TorchOSQPWorkspace()
    solve_torch_osqp_direct(
        P, P.new_tensor([-0.25]), A, l, u, settings, first_workspace
    )
    _, second_info = solve_torch_osqp_direct(
        P, P.new_tensor([-0.75]), A, l, u, settings, second_workspace
    )
    second_state = {
        key: value.clone() for key, value in second_workspace.state.items()
    }
    second_counts = (
        second_workspace.linear_solver.factorization_count,
        second_workspace.linear_solver.solve_count,
    )

    solve_torch_osqp_direct(
        P, P.new_tensor([-0.5]), A, l, u, settings, first_workspace
    )

    assert first_workspace.linear_solver is not second_workspace.linear_solver
    assert first_workspace.state is not second_workspace.state
    assert second_info["factorization_count"] == second_counts[0]
    assert (
        second_workspace.linear_solver.factorization_count,
        second_workspace.linear_solver.solve_count,
    ) == second_counts
    for key, value in second_state.items():
        assert torch.equal(second_workspace.state[key], value)


def test_dense_to_sparse_input_reuses_canonical_dense_workspace(
    floating_dtype, available_device, monkeypatch
):
    from pygranso.private.osqpTorchAdapter import solve_osqp_torch_qp

    dtype = floating_dtype
    device = available_device
    H = torch.eye(2, dtype=dtype, device=device)
    f = torch.tensor([[-1.0], [-0.5]], dtype=dtype, device=device)
    lower = torch.zeros((2, 1), dtype=dtype, device=device)
    upper = torch.ones((2, 1), dtype=dtype, device=device)
    options = {
        "algebra": "torch",
        "settings": {
            "scaling": 0,
            "adaptive_rho": False,
            "polishing": False,
            "check_termination": 1,
            "return_info": True,
        },
    }
    workspace = TorchOSQPWorkspace()
    solve_osqp_torch_qp(
        H, f, None, None, lower, upper, device, dtype == torch.float64,
        options, workspace
    )
    state_before = workspace.state
    observed = {}
    original = torch_osqp_module._initial_scaled_state

    def capture_warm_state(active_workspace, *args, **kwargs):
        observed["preserved"] = active_workspace.state is state_before
        return original(active_workspace, *args, **kwargs)

    monkeypatch.setattr(
        torch_osqp_module, "_initial_scaled_state", capture_warm_state
    )
    with pytest.warns(RuntimeWarning, match="convert sparse input"):
        _, info = solve_osqp_torch_qp(
            H.to_sparse(),
            f,
            None,
            None,
            lower,
            upper,
            device,
            dtype == torch.float64,
            options,
            workspace,
        )

    assert observed == {"preserved": True}
    assert info["factorization_reused"]
    assert info["factorizations_this_solve"] == 0
    assert info["sparse_to_dense"]["occurred"]
    assert set(info["sparse_to_dense"]["inputs"]) == {"H"}


def test_a_value_update_preserves_state_but_forces_new_factors():
    from pygranso.private.osqpTorchAdapter import solve_osqp_torch_qp

    dtype = torch.float64
    workspace = TorchOSQPWorkspace()
    H = torch.eye(2, dtype=dtype)
    f = torch.zeros((2, 1), dtype=dtype)
    lower = -torch.ones((2, 1), dtype=dtype)
    upper = torch.ones((2, 1), dtype=dtype)
    settings = {"return_info": True, "polishing": False}
    _, first = solve_osqp_torch_qp(
        H,
        f,
        torch.tensor([[1.0, 1.0]], dtype=dtype),
        torch.tensor(0.0, dtype=dtype),
        lower,
        upper,
        "cpu",
        True,
        {"algebra": "torch", "settings": settings},
        workspace,
    )
    prior_state = {key: value.clone() for key, value in workspace.state.items()}
    _, second = solve_osqp_torch_qp(
        H,
        f,
        torch.tensor([[2.0, 1.0]], dtype=dtype),
        torch.tensor(0.0, dtype=dtype),
        lower,
        upper,
        "cpu",
        True,
        {"algebra": "torch", "settings": settings},
        workspace,
    )
    assert first["factorizations_this_solve"] >= 1
    assert second["factorizations_this_solve"] >= 1
    assert prior_state.keys() == workspace.state.keys()


def _h07_first_step(rho, state):
    """Replay the scalar pinned vector-rho path, with a rational KKT solve."""
    x, z, y = state
    rho_inv = 1.0 / rho
    matrix = [[1.0625, 1.0], [1.0, -rho_inv]]
    rhs = [0.0625 * x + 1.0, z - rho_inv * y]
    a, d, b0, b1 = map(Fraction, (matrix[0][0], matrix[1][1], *rhs))
    x_tilde = float((d * b0 - b1) / (a * d - 1))
    nu = float((a * b1 - b0) / (a * d - 1))
    z_tilde = rhs[1] + rho_inv * nu
    z_next = min(max(rho_inv * y + z_tilde, 0.0), 0.25)
    delta_y = rho * (z_tilde - z_next)
    return {
        "initial": list(state), "rho_vector": [rho], "rho_inverse": [rho_inv],
        "kkt": matrix, "rhs": rhs, "linear_solution": [x_tilde, nu],
        "z_tilde": z_tilde, "updated": [x_tilde, z_next, y + delta_y],
        "delta_y": delta_y,
    }


def _h07_solve(workspace, **settings):
    H = torch.ones((1, 1), dtype=torch.float64)
    q = -H.reshape(-1)
    return solve_osqp_torch_qp(
        H, q, None, None, torch.zeros_like(q), H.reshape(-1) * 0.25,
        H.device, True,
        {"algebra": "torch", "settings": {
            "max_iter": 1, "check_termination": 0, "adaptive_rho": 1,
            "rho_update_interval": 1, "rho_update_tolerance": 2.0,
            "rho_update_safeguard": False, "scaling": 0, "sigma": 0.0625,
            "alpha": 1.0, "eps_abs": 0.125, "eps_rel": 0.0, "return_info": True,
            **settings,
        }},
        workspace,
    )


@pytest.mark.parametrize(
    "case", ["cold", "warm", "fresh", "no-adaptation", "explicit-rho", "rollback"]
)
def test_h07_first_kkt_after_workspace_cold_start(
    case, monkeypatch, record_property
):
    workspace = TorchOSQPWorkspace()
    solver = workspace.linear_solver
    steps = []
    rho_attempts = []
    adapt = torch_osqp_module._adaptive_rho_update
    rhs_builder = torch_osqp_module.build_kkt_rhs
    solve = type(solver).solve
    update = torch_osqp_module.admm_vector_update

    def capture_rhs(x, z, y, q, sigma, rho_inv_vec):
        result = rhs_builder(x, z, y, q, sigma, rho_inv_vec)
        steps.append({
            "initial": [v.item() for v in (x, z, y)],
            "rho_inverse": rho_inv_vec.tolist(),
            "rhs": result.tolist(),
        })
        return result

    def capture_solve(solver, rhs, **kwargs):
        solution, info = solve(solver, rhs, **kwargs)
        steps[-1].update(kkt=solver.matrix.tolist(), linear_solution=solution.tolist())
        return solution, info

    def capture_update(x_tilde, x, z_tilde, z, y, rho_vec, *args, **kwargs):
        result = update(x_tilde, x, z_tilde, z, y, rho_vec, *args, **kwargs)
        steps[-1].update(
            rho_vector=rho_vec.tolist(),
            z_tilde=z_tilde.item(),
            updated=[v.item() for v in result[:3]],
            delta_y=result[3].item(),
        )
        return result

    def capture_adapt(*args, **kwargs):
        result = adapt(*args, **kwargs)
        rho_attempts.append(result[0])
        return result

    monkeypatch.setattr(torch_osqp_module, "_adaptive_rho_update", capture_adapt)
    monkeypatch.setattr(torch_osqp_module, "build_kkt_rhs", capture_rhs)
    monkeypatch.setattr(type(solver), "solve", capture_solve)
    monkeypatch.setattr(torch_osqp_module, "admm_vector_update", capture_update)

    requested_rho = 0.1
    mode = 0 if case == "no-adaptation" else 1
    _, first = _h07_solve(workspace, adaptive_rho=mode)
    assert first["status"] == "solved_inaccurate"
    assert len(steps) == 1
    cold_state = (0.0, 0.0, 0.0)
    first_reference = _h07_first_step(requested_rho, cold_state)
    x, z, y = first_reference["updated"]
    # auxil.c::compute_rho_estimate uses the separately normalized residuals.
    prim = abs(x - z) / (max(abs(x), abs(z)) + 1e-30)
    dual = abs(-1.0 + x + y) / (max(1.0, abs(x), abs(y)) + 1e-30)
    candidate = requested_rho * math.sqrt(prim / dual)
    assert candidate > 2.0 * requested_rho
    assert first["rho_updates"] == int(mode == 1)
    if mode:
        assert workspace.rho_bar == pytest.approx(candidate, rel=1e-14)
    else:
        assert workspace.rho_bar == requested_rho
    assert workspace.rho_setting == requested_rho
    assert first["factorizations_this_solve"] == 1 + int(mode == 1)
    assert workspace.linear_solver is solver
    assert workspace.linear_solver.matrix[-1, -1].item() == -1.0 / workspace.rho_bar
    committed = tuple(workspace.state[key].item() for key in ("x", "z", "y"))
    assert committed == pytest.approx(first_reference["updated"], rel=1e-14)
    assert all(committed)
    for metric, expected in first_reference.items():
        observed = steps[0][metric]
        if metric in {"initial", "kkt", "rho_vector", "rho_inverse", "rhs"}:
            assert observed == expected
        else:
            assert observed == pytest.approx(expected, rel=1e-14, abs=1e-15)
    first_id = id(workspace)
    first_setup = {
        "requested_rho": requested_rho, "candidate": candidate,
        "active_rho": workspace.rho_bar, "rho_vector": [workspace.rho_bar],
        "rho_inverse": [1.0 / workspace.rho_bar], "rho_updates": first["rho_updates"],
        "status": first["status"], "factors": first["factorizations_this_solve"],
        "kkt": solver.matrix.tolist(), "state": list(committed), "workspace_id": first_id,
        "solver_id": id(solver), "constraint_types": workspace.constraint_types.tolist(),
        "rho_setup_signature": workspace.rho_setup_signature,
        "effective_update_interval": workspace.adaptive_rho_interval,
    }
    target = TorchOSQPWorkspace() if case == "fresh" else workspace
    second_settings = {
        "adaptive_rho": mode, "warm_start": case == "warm",
        "check_termination": 1, "eps_abs": 2.0,
    }
    if case == "explicit-rho":
        second_settings["rho"] = 0.25
    if case == "rollback":
        fields_before = workspace.__dict__.copy()
        factorize = solver.factorize
        attempts = []

        def fail_after_factorize(matrix, **kwargs):
            attempts.append(matrix.tolist())
            factorize(matrix, **kwargs)
            raise TorchLinearSolveError("failed explicit rho refactorization")

        with monkeypatch.context() as patch:
            patch.setattr(solver, "factorize", fail_after_factorize)
            solver_before = solver.__dict__.copy()
            with pytest.raises(TorchLinearSolveError, match="failed explicit rho"):
                _h07_solve(target, **second_settings, rho=0.25)
            for key, value in solver_before.items():
                assert solver.__dict__[key] is value
        assert attempts == [[[1.0625, 1.0], [1.0, -4.0]]]
        for key, value in fields_before.items():
            if key == "builtin_stats":
                assert workspace.builtin_stats == value
            else:
                assert workspace.__dict__[key] is value
        assert len(steps) == 1
        record_property("failed_refactor_kkt", json.dumps(attempts[0]))

    before_attempts = len(rho_attempts)
    _, second = _h07_solve(target, **second_settings)
    expected_rho = {
        "fresh": requested_rho, "explicit-rho": 0.25,
    }.get(case, first_setup["active_rho"])
    expected_state = committed if case == "warm" else cold_state
    second_reference = _h07_first_step(expected_rho, expected_state)
    record_property("first_post_solve", json.dumps(first_setup))
    record_property("expected_first_step", json.dumps(first_reference))
    record_property("expected_second_step", json.dumps(second_reference))
    record_property("observed_steps", json.dumps(steps))
    record_property("second_post_solve", json.dumps({
        "workspace_id": id(target), "solver_id": id(target.linear_solver),
        "requested_rho": target.rho_setting, "active_rho": target.rho_bar,
        "status": second["status"], "rho_updates": second["rho_updates"],
        "factors": second["factorizations_this_solve"],
        "factorization_reused": second["factorization_reused"],
        "kkt": target.linear_solver.matrix.tolist(),
        "state": [target.state[key].item() for key in ("x", "z", "y")],
        "constraint_types": target.constraint_types.tolist(),
        "rho_setup_signature": target.rho_setup_signature,
        "effective_update_interval": target.adaptive_rho_interval,
    }))
    assert len(steps) == 2
    assert second["status"] == "solved"
    assert rho_attempts[before_attempts:] == []  # Exact termination precedes adaptation.
    assert second["rho_updates"] == (0 if case == "fresh" else first["rho_updates"])
    assert steps[1]["initial"] == list(expected_state)
    assert target.rho_bar == expected_rho
    assert target.rho_setting == second_settings.get("rho", requested_rho)
    for metric, expected in second_reference.items():
        observed = steps[1][metric]
        if metric in {"initial", "kkt", "rho_vector", "rho_inverse"}:
            assert observed == expected
        else:
            assert observed == pytest.approx(expected, rel=1e-14, abs=1e-15)
    assert [target.state[k].item() for k in ("x", "z", "y")] == steps[1]["updated"]
    reuse = case not in {"fresh", "explicit-rho"}
    assert second["factorization_reused"] == reuse
    assert second["factorizations_this_solve"] == int(not reuse)
    assert (target.linear_solver is solver) == (case != "fresh")
    if case in {"cold", "rollback"}:
        reset = _h07_first_step(requested_rho, cold_state)
        assert abs(steps[1]["updated"][0] - reset["updated"][0]) > 0.01
