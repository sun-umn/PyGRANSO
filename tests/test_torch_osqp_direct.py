import json

import pytest
import torch

from pygranso.private import torchOSQP as torch_osqp_module
from pygranso.private.osqpTorchAdapter import DEFAULT_OSQP_SETTINGS, solve_osqp_torch_qp
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchLinearSolve import TorchLinearSolveError


def _bound_problem(dtype=torch.float64, device="cpu"):
    device = torch.device(device)
    H = torch.eye(2, device=device, dtype=dtype)
    f = torch.tensor([[-1.0], [-2.0]], device=device, dtype=dtype)
    lower = torch.zeros((2, 1), device=device, dtype=dtype)
    upper = torch.ones((2, 1), device=device, dtype=dtype)
    return H, f, None, None, lower, upper


def _solve(problem, *, settings=None, workspace=None, algebra="torch"):
    H, f, A, b, lower, upper = problem
    options = {"algebra": algebra, "settings": {"return_info": True}}
    options["settings"].update(settings or {})
    return solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        lower,
        upper,
        f.device,
        f.dtype == torch.float64,
        options,
        workspace,
    )


def test_bound_qp_device_dtype_shape_and_solution(floating_dtype, available_device):
    solution, info = _solve(_bound_problem(floating_dtype, available_device))
    tolerance = 3e-4 if floating_dtype == torch.float32 else 1e-7
    assert solution.shape == (2, 1)
    assert solution.device.type == available_device.type
    if available_device.index is not None:
        assert solution.device.index == available_device.index
    assert solution.dtype == floating_dtype
    assert torch.allclose(
        solution.reshape(-1),
        torch.tensor([1.0, 1.0], device=available_device, dtype=floating_dtype),
        atol=tolerance,
        rtol=tolerance,
    )
    assert info["status"] == "solved"
    assert info["primal_residual"] <= info["eps_primal"]
    assert info["dual_residual"] <= info["eps_dual"]
    assert info["polishing"] is False
    assert info["polishing_status"] == "disabled"


def test_time_limit_stops_after_a_completed_iteration_and_keeps_latest_iterate(
    monkeypatch, floating_dtype, available_device
):
    P = torch.eye(2, device=available_device, dtype=floating_dtype)
    q = torch.tensor([-1.0, -2.0], device=available_device, dtype=floating_dtype)
    A = torch.eye(2, device=available_device, dtype=floating_dtype)
    l = torch.zeros(2, device=available_device, dtype=floating_dtype)
    u = torch.ones(2, device=available_device, dtype=floating_dtype)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "time_limit": 0.5,
        "max_iter": 20,
        "check_termination": 0,
        "adaptive_rho": False,
        "polishing": True,
        "scaling": 0,
        "eps_abs": 1e-12,
        "eps_rel": 1e-12,
        "return_info": True,
    }
    clock_calls = 0

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls == 1 else 0.5

    monkeypatch.setattr(torch_osqp_module, "_clock", clock)
    solution, info = torch_osqp_module.solve_torch_osqp_direct(
        P, q, A, l, u, settings, TorchOSQPWorkspace()
    )

    assert info["status"] == "time_limit_reached"
    assert info["status_val"] == 8
    assert not info["status_compatible"]
    assert info["admm_iterations"] == 0
    assert info["linear_solves_this_solve"] == 1
    assert torch.all(torch.isfinite(solution))
    assert info["primal_infeasibility_certificate"] is None
    assert info["dual_infeasibility_certificate"] is None
    assert info["polishing_status"] == "skipped_unsolved"


def test_time_limit_precedes_periodic_termination_and_adaptive_rho(monkeypatch):
    P = torch.eye(2, dtype=torch.float64)
    q = torch.tensor([-1.0, -2.0], dtype=torch.float64)
    A = torch.eye(2, dtype=torch.float64)
    l = torch.zeros(2, dtype=torch.float64)
    u = torch.ones(2, dtype=torch.float64)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "time_limit": 0.5,
        "max_iter": 2,
        "check_termination": 1,
        "adaptive_rho": True,
        "rho_update_interval": 1,
        "polishing": False,
        "scaling": 0,
        "return_info": True,
    }
    clock_calls = 0
    termination_checks = []

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls == 1 else 0.5

    def no_termination(*args, approximate, **kwargs):
        termination_checks.append(approximate)
        return None

    def unexpected_rho_update(*args, **kwargs):
        raise AssertionError("rho update must not run after the time limit is reached")

    monkeypatch.setattr(torch_osqp_module, "_clock", clock)
    monkeypatch.setattr(torch_osqp_module, "_termination_status", no_termination)
    monkeypatch.setattr(
        torch_osqp_module, "_adaptive_rho_update", unexpected_rho_update
    )

    _, info = torch_osqp_module.solve_torch_osqp_direct(
        P, q, A, l, u, settings, TorchOSQPWorkspace()
    )

    assert info["status"] == "time_limit_reached"
    assert info["admm_iterations"] == 0
    assert info["linear_solves_this_solve"] == 1
    assert termination_checks == [False, True]


def test_time_limit_final_inaccurate_check_can_replace_timeout(monkeypatch):
    P = torch.eye(1, dtype=torch.float64)
    q = torch.tensor([-1.0], dtype=torch.float64)
    A = torch.eye(1, dtype=torch.float64)
    l = torch.zeros(1, dtype=torch.float64)
    u = torch.ones(1, dtype=torch.float64)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "time_limit": 0.5,
        "max_iter": 2,
        "check_termination": 0,
        "adaptive_rho": False,
        "polishing": False,
        "scaling": 0,
        "return_info": True,
    }
    clock_calls = 0
    termination_checks = []

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls == 1 else 0.5

    def inaccurate_only(*args, approximate, **kwargs):
        termination_checks.append(approximate)
        if approximate:
            return "solved_inaccurate", None, None
        return None

    monkeypatch.setattr(torch_osqp_module, "_clock", clock)
    monkeypatch.setattr(torch_osqp_module, "_termination_status", inaccurate_only)

    _, info = torch_osqp_module.solve_torch_osqp_direct(
        P, q, A, l, u, settings, TorchOSQPWorkspace()
    )

    assert info["status"] == "solved_inaccurate"
    assert info["admm_iterations"] == 0
    assert info["linear_solves_this_solve"] == 1
    assert termination_checks == [False, True]


def _timeout_trace(monkeypatch, record_property, settings, problem, last_update):
    ticks = [0.0, 0.0, *([0.25] * (last_update - 1)), 0.5]
    now = iter(ticks)
    monkeypatch.setattr(torch_osqp_module, "_clock", lambda: next(now))
    original = torch_osqp_module._termination_status
    checks = []

    def observe(*args, approximate, objective_values, **kwargs):
        decision = original(
            *args, approximate=approximate, objective_values=objective_values, **kwargs
        )
        checks.append(
            {
                "mode": "approximate" if approximate else "exact",
                "residuals_and_thresholds": [float(v) for v in args[7]],
                "objective_and_gap": {
                    k: float(objective_values[k]) for k in (
                        "objective", "dual_objective", "duality_gap", "eps_duality_gap"
                    )
                },
                "delta_y": kwargs["primal_certificate_context"][3].tolist(),
                "decision": None if decision is None else decision[0],
            }
        )
        return decision

    monkeypatch.setattr(torch_osqp_module, "_termination_status", observe)
    solution, info = torch_osqp_module.solve_torch_osqp_direct(
        *problem, settings, TorchOSQPWorkspace()
    )
    record_property("ticks", json.dumps(ticks))
    record_property("checks", json.dumps(checks))
    record_property(
        "result",
        json.dumps(
            {
                **{
                    key: info[key]
                    for key in (
                        "status", "status_val", "admm_iterations",
                        "linear_solves_this_solve", "primal_residual", "dual_residual",
                        "eps_primal", "eps_dual", "objective", "dual_objective",
                        "duality_gap", "eps_duality_gap", "rel_kkt_error", "rho_estimate",
                    )
                },
                "x": solution.tolist(),
                "primal_certificate": (
                    None if info["prim_inf_cert"] is None else info["prim_inf_cert"].tolist()
                ),
            }
        ),
    )
    assert next(now, None) is None
    assert info["linear_solves_this_solve"] == last_update
    assert info["rho_updates"] == 0
    return solution, info, checks


def _timeout_settings(**updates):
    return {
        **DEFAULT_OSQP_SETTINGS,
        "time_limit": 0.5,
        "max_iter": 8,
        "alpha": 1.0,
        "sigma": 1.0,
        "scaling": 0,
        "adaptive_rho": False,
        "warm_start": False,
        "polishing": False,
        "check_dualgap": True,
        "eps_abs": 0.0625,
        "eps_rel": 0.125,
        "return_info": True,
        **updates,
    }


@pytest.mark.parametrize(
    "interval,last_update,eps_abs,eps_rel,decisions,expected",
    [
        pytest.param(
            2, 3, 0.0625, 0.125, [None, "solved_inaccurate"],
            ("solved_inaccurate", 2, 0.25, -0.546875, -0.328125, -0.21875, 0.25),
            id="checked-predecessor",
        ),
        pytest.param(
            2, 4, 0.0625, 0.125, [None, "solved"],
            ("solved", 1, 0.0625, -0.498046875, -0.439453125, -0.05859375, 0.0625),
            id="unchecked-predecessor",
        ),
        pytest.param(
            1, 1, 0.0625, 0.125, [None, "solved_inaccurate"],
            ("solved_inaccurate", 2, 0.5, -0.375, -0.125, -0.25, 0.5),
            id="first-iteration",
        ),
        pytest.param(
            0, 3, 0.0625, 0.125, ["solved"],
            ("solved", 1, 0.125, -0.4921875, -0.3828125, -0.109375, 0.125),
            id="disabled-periodic-exact-footer",
        ),
        pytest.param(
            0, 3, 0.0078125, 0.0, [None, None],
            ("time_limit_reached", 8, 0.125, -0.4921875, -0.3828125, -0.109375, 0.125),
            id="disabled-periodic-restore-timeout",
        ),
    ],
)
def test_e06_timeout_footer_order_and_info(
    monkeypatch, record_property, interval, last_update, eps_abs, eps_rel, decisions, expected
):
    dtype = torch.float64
    # With P=1, q=-1, sigma=1 and alpha=1, the iterate is x_k=1-2**(-k).
    problem = (
        torch.ones((1, 1), dtype=dtype), -torch.ones(1, dtype=dtype),
        torch.empty((0, 1), dtype=dtype), torch.empty(0, dtype=dtype),
        torch.empty(0, dtype=dtype),
    )
    settings = _timeout_settings(
        check_termination=interval, eps_abs=eps_abs, eps_rel=eps_rel
    )
    solution, info, checks = _timeout_trace(
        monkeypatch, record_property, settings, problem, last_update
    )
    latest_x = 1 - 2.0**-last_update
    assert solution.item() == latest_x
    assert tuple(info[k] for k in (
        "status", "status_val", "dual_residual", "objective", "dual_objective",
        "duality_gap", "rel_kkt_error",
    )) == expected
    # The timeout exits via break, so the source records iter-1 even when
    # update_info actually reads the latest ADMM state.
    assert info["admm_iterations"] == last_update - 1
    assert [c["decision"] for c in checks] == decisions
    modes = ["exact"] * len(decisions)
    if expected[1] in (2, 8):
        modes[-1] = "approximate"
    assert [c["mode"] for c in checks] == modes
    assert info["eps_dual"] == eps_abs + eps_rel
    assert info["eps_duality_gap"] == eps_abs + eps_rel * latest_x
    for check, mode in zip(checks, modes, strict=True):
        scale = 10.0 if mode == "approximate" else 1.0
        assert check["residuals_and_thresholds"] == [
            0.0, (0.25 if check is checks[0] and interval == 2 else expected[2]),
            eps_abs * scale, eps_abs * scale + (eps_rel * scale),
        ]
    assert checks[-1]["objective_and_gap"]["duality_gap"] == expected[-2]


def test_e06_timeout_checked_predecessor_mixes_cached_products_with_live_state(
    monkeypatch, record_property
):
    dtype = torch.float64
    problem = (
        torch.ones((1, 1), dtype=dtype), -torch.ones(1, dtype=dtype),
        torch.ones((1, 1), dtype=dtype), torch.zeros(1, dtype=dtype),
        torch.ones(1, dtype=dtype),
    )
    updates = iter(((0.5, 0.25, 0.0, 0.0), (2.0, 1.0, 3.5, 3.5)))
    monkeypatch.setattr(
        torch_osqp_module, "admm_vector_update",
        lambda *args, **kwargs: tuple(torch.tensor([v], dtype=dtype) for v in next(updates)),
    )
    solution, info, checks = _timeout_trace(
        monkeypatch, record_property, _timeout_settings(check_termination=1), problem, 2
    )
    assert solution.item() == 2.0
    assert (info["status"], info["status_val"], info["admm_iterations"]) == (
        "solved_inaccurate", 2, 1
    )
    assert [(c["mode"], c["decision"]) for c in checks] == [
        ("exact", None), ("approximate", "solved_inaccurate")
    ]
    assert checks[0]["residuals_and_thresholds"] == [0.25, 0.5, 0.125, 0.1875]
    # Cached Ax=0.5, Px=0.5, A'y=0; live x=2, z=1, y=3.5.
    assert checks[1]["residuals_and_thresholds"] == [0.25, 0.5, 1.875, 1.875]
    assert checks[1]["objective_and_gap"] == {
        "objective": -1.5, "dual_objective": -4.0, "duality_gap": 2.5,
        "eps_duality_gap": 5.0,
    }
    assert checks[1]["delta_y"] == [3.5]
    assert info["primal_residual"] == 0.25
    assert info["dual_residual"] == info["rel_kkt_error"] == 0.5
    assert info["eps_primal"] == info["eps_dual"] == 0.1875
    assert info["eps_duality_gap"] == 0.5
    assert info["rho_estimate"] == pytest.approx(0.1 * (0.5**0.5), rel=1e-14)


@pytest.mark.parametrize("last_update", [1, 2], ids=["exact", "approximate"])
def test_e06_timeout_infeasibility_replaces_timeout_with_latest_increment(
    monkeypatch, record_property, last_update
):
    dtype = torch.float64
    zero = torch.zeros(1, dtype=dtype)
    bounds = torch.tensor([1.0, 0.0], dtype=dtype)
    certificate = torch.tensor([-1.0, 1.0], dtype=dtype)
    updates = iter(
        [(zero, bounds, torch.zeros_like(bounds), torch.zeros_like(bounds))]
        * (last_update - 1)
        + [(zero, bounds, certificate, certificate)]
    )
    monkeypatch.setattr(
        torch_osqp_module, "admm_vector_update", lambda *args, **kwargs: next(updates)
    )
    problem = (
        torch.ones((1, 1), dtype=dtype), zero,
        torch.ones((2, 1), dtype=dtype), bounds, bounds,
    )
    _, info, checks = _timeout_trace(
        monkeypatch, record_property,
        _timeout_settings(
            check_termination=1, eps_abs=0.01, eps_rel=0.0, check_dualgap=False
        ),
        problem, last_update,
    )
    expected_status = ("primal_infeasible", "primal_infeasible_inaccurate")[last_update - 1]
    assert (info["status"], info["status_val"], info["admm_iterations"]) == (
        expected_status, last_update + 2, last_update - 1
    )
    assert [c["mode"] for c in checks] == ["exact", "approximate"][:last_update]
    assert checks[-1]["delta_y"] == [-1.0, 1.0]
    assert checks[-1]["decision"] == expected_status
    assert torch.equal(info["prim_inf_cert"], certificate)
    assert info["dual_inf_cert"] is None
    assert info["objective"] == 1e30


def test_equality_plus_bounds_and_infinite_bound():
    dtype = torch.float64
    H = torch.eye(2, dtype=dtype)
    f = torch.zeros((2, 1), dtype=dtype)
    A = torch.tensor([[1.0, 1.0]], dtype=dtype)
    b = torch.tensor(1.0, dtype=dtype)
    lower = torch.tensor([[-torch.inf], [0.0]], dtype=dtype)
    upper = torch.tensor([[torch.inf], [1.0]], dtype=dtype)
    solution, info = _solve((H, f, A, b, lower, upper))
    assert torch.allclose(solution.sum(), torch.tensor(1.0, dtype=dtype), atol=1e-7)
    assert info["status"] == "solved"


def test_noncontiguous_views_are_not_mutated_by_scaling_or_solve():
    dtype = torch.float64
    H = torch.eye(2, dtype=dtype).T
    f = torch.tensor([-1.0, 9.0, -2.0, 9.0], dtype=dtype)[::2].reshape(2, 1)
    A = torch.tensor([[1.0], [1.0]], dtype=dtype).T
    b = torch.tensor(1.0, dtype=dtype)
    lower = torch.tensor([0.0, 9.0, 0.0, 9.0], dtype=dtype)[::2].reshape(2, 1)
    upper = torch.tensor([1.0, 9.0, 1.0, 9.0], dtype=dtype)[::2].reshape(2, 1)
    inputs = (H, f, A, b, lower, upper)
    originals = tuple(value.clone() for value in inputs)
    solution, info = _solve(inputs)
    assert info["status_compatible"]
    assert torch.all(torch.isfinite(solution))
    for value, original in zip(inputs, originals):
        assert torch.equal(value, original)


def test_warm_start_and_lu_reuse_for_vector_update():
    workspace = TorchOSQPWorkspace()
    problem = _bound_problem()
    reuse_settings = {"polishing": False, "adaptive_rho": False}
    _, first = _solve(problem, settings=reuse_settings, workspace=workspace)
    changed = list(problem)
    changed[1] = torch.tensor([[-0.5], [-1.5]], dtype=torch.float64)
    _, second = _solve(
        tuple(changed), settings=reuse_settings, workspace=workspace
    )
    assert first["factorizations_this_solve"] >= 1
    assert second["factorizations_this_solve"] == 0
    assert second["factorization_reused"]


def test_matrix_value_update_refactorizes_but_keeps_workspace_state():
    workspace = TorchOSQPWorkspace()
    problem = _bound_problem()
    _solve(problem, settings={"polishing": False}, workspace=workspace)
    changed = list(problem)
    changed[0] = 2.0 * changed[0]
    _solve(tuple(changed), settings={"polishing": False}, workspace=workspace)
    assert workspace.state is not None
    assert workspace.linear_solver.factorization_count >= 2


def test_scaling_adaptive_rho_and_polishing_are_reported():
    _, info = _solve(_bound_problem(), settings={"polishing": True})
    assert info["scaling_applied"]
    assert info["scaling_passes"] == 10
    assert "rho_updates" in info
    assert info["polishing_status"] == "accepted"


def test_requested_rho_change_refactorizes(monkeypatch):
    workspace = TorchOSQPWorkspace()
    problem = _bound_problem()
    _solve(problem, settings={"polishing": False, "rho": 0.1}, workspace=workspace)
    state_before = workspace.state
    observed = {}
    original = torch_osqp_module._initial_scaled_state

    def capture_warm_state(active_workspace, *args, **kwargs):
        observed["preserved"] = active_workspace.state is state_before
        return original(active_workspace, *args, **kwargs)

    monkeypatch.setattr(
        torch_osqp_module, "_initial_scaled_state", capture_warm_state
    )
    _, info = _solve(
        problem,
        settings={"polishing": False, "rho": 0.2},
        workspace=workspace,
    )
    assert observed == {"preserved": True}
    assert info["factorizations_this_solve"] >= 1
    assert workspace.rho_setting == 0.2


@pytest.mark.parametrize("requested,expected", [(1e-20, 1e-6), (1e20, 1e6)])
def test_requested_rho_is_clamped_to_pinned_setup_bounds(requested, expected):
    workspace = TorchOSQPWorkspace()
    try:
        _, info = _solve(
            _bound_problem(),
            settings={"polishing": False, "adaptive_rho": False, "rho": requested},
            workspace=workspace,
        )
    except RuntimeError as exc:
        assert "max_iter_reached" in str(exc)
        info = workspace.last_info
    assert info["rho_bar"] == expected


def test_requested_sigma_change_refactorizes(monkeypatch):
    workspace = TorchOSQPWorkspace()
    problem = _bound_problem()
    _solve(problem, settings={"polishing": False, "sigma": 1e-6}, workspace=workspace)
    state_before = workspace.state
    observed = {}
    original = torch_osqp_module._initial_scaled_state

    def capture_warm_state(active_workspace, *args, **kwargs):
        observed["preserved"] = active_workspace.state is state_before
        return original(active_workspace, *args, **kwargs)

    monkeypatch.setattr(
        torch_osqp_module, "_initial_scaled_state", capture_warm_state
    )
    _, info = _solve(
        problem,
        settings={"polishing": False, "sigma": 1e-5},
        workspace=workspace,
    )
    assert observed == {"preserved": True}
    assert info["factorizations_this_solve"] >= 1
    assert not info["factorization_reused"]


def test_incompatible_requested_dtype_is_rejected():
    problem = _bound_problem(torch.float32)
    with pytest.raises(ValueError, match="requested precision"):
        solve_osqp_torch_qp(
            *problem,
            torch.device("cpu"),
            True,
            {"algebra": "torch"},
            TorchOSQPWorkspace(),
        )


def test_input_validation_rejects_invalid_problem():
    problem = list(_bound_problem())
    problem[0] = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="asymmetric"):
        _solve(tuple(problem))

    problem = list(_bound_problem())
    problem[4] = torch.ones((2, 1), dtype=torch.float64)
    problem[5] = torch.zeros((2, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match="lower bound"):
        _solve(tuple(problem))

    problem = list(_bound_problem())
    problem[0][0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        _solve(tuple(problem))


def test_pinned_vector_all_leq_patterns_match_bound_validation(
    floating_dtype, available_device
):
    """Translate every all-leq assertion in test_vector_comparisons.cpp."""
    v1 = torch.tensor(
        [
            -0.31389947196684775,
            0.05410227877154389,
            0.27279133916445375,
            -0.9821881249409777,
            -1.107373047165193,
            0.19958453284708083,
            -0.46674961687980204,
            0.23550561173022522,
            0.7595195224783792,
            -1.6487873663509485,
        ],
        device=available_device,
        dtype=floating_dtype,
    )
    v2 = torch.tensor(
        [
            0.2543881165176173,
            1.2246469675357323,
            -0.2975268443704732,
            -0.8108145832375699,
            0.7522438271795928,
            0.25344651620814146,
            0.8958830707775604,
            -0.3452157100512797,
            -1.4818182737222112,
            -0.11001076471125099,
        ],
        device=available_device,
        dtype=floating_dtype,
    )
    zero = torch.zeros_like(v1)
    same = torch.full_like(v1, 0.4684043358472779)
    shifted_v1 = v1 + 4.0
    shifted_v2 = -v2 - 4.0
    cases = (
        ("negated-vector", -v1, v1, False),
        ("zero-before-positive-constant", zero, same, True),
        ("positive-constant-before-zero", same, zero, False),
        ("positive-shift-before-original", shifted_v1, v1, False),
        ("original-before-positive-shift", v1, shifted_v1, True),
        ("opposed-shifts-forward", shifted_v1, shifted_v2, False),
        ("opposed-shifts-reverse", shifted_v2, shifted_v1, True),
    )
    P = torch.eye(v1.numel(), device=available_device, dtype=floating_dtype)
    q = torch.zeros_like(v1)
    A = P.clone()

    for case_name, lower, upper, expected in cases:
        if expected:
            _, _, _, validated_lower, validated_upper = torch_osqp_module._validate_qp(
                P, q, A, lower, upper, {}
            )
            assert torch.equal(validated_lower, lower), case_name
            assert torch.equal(validated_upper, upper), case_name
        else:
            with pytest.raises(ValueError, match="lower bound"):
                torch_osqp_module._validate_qp(P, q, A, lower, upper, {})


def test_input_validation_rejects_zero_variables_and_malformed_shapes():
    dtype = torch.float64
    with pytest.raises(ValueError, match="at least one variable"):
        solve_osqp_torch_qp(
            torch.empty((0, 0), dtype=dtype),
            torch.empty((0, 1), dtype=dtype),
            None,
            None,
            torch.empty((0, 1), dtype=dtype),
            torch.empty((0, 1), dtype=dtype),
            "cpu",
            True,
            {"algebra": "torch"},
            TorchOSQPWorkspace(),
        )

    problem = list(_bound_problem())
    problem[0] = torch.eye(3, dtype=dtype)
    with pytest.raises(ValueError, match=r"must have shape \(2, 2\)"):
        _solve(tuple(problem))

    problem = list(_bound_problem())
    problem[2] = torch.ones((1, 3), dtype=dtype)
    problem[3] = torch.zeros((1, 1), dtype=dtype)
    with pytest.raises(ValueError, match="must have 2 columns"):
        _solve(tuple(problem))

    problem = list(_bound_problem())
    problem[4] = torch.zeros((3, 1), dtype=dtype)
    with pytest.raises(ValueError, match=r"LB and UB must have len\(f\) entries"):
        _solve(tuple(problem))

    problem = list(_bound_problem())
    problem[2] = torch.ones((2, 2), dtype=dtype)
    problem[3] = torch.zeros((3, 1), dtype=dtype)
    with pytest.raises(ValueError, match="one entry per row of A"):
        _solve(tuple(problem))

    problem = list(_bound_problem())
    problem[0] = None
    with pytest.raises(TypeError, match="H must be a Torch tensor"):
        _solve(tuple(problem))


@pytest.mark.parametrize(
    "case,error_type,match",
    [
        ("missing_p", TypeError, "P must be a Torch tensor"),
        ("missing_q", TypeError, "q must be a Torch tensor"),
        ("missing_a", TypeError, "A must be a Torch tensor"),
        ("zero_variables", ValueError, "at least one variable"),
        ("p_shape", ValueError, "P must have shape"),
        ("a_shape", ValueError, "A must have shape"),
        ("bound_length", ValueError, "same number of entries"),
        ("lower_triangle", ValueError, "materially asymmetric"),
        ("nonfinite_p", ValueError, "P contains NaN or Inf"),
        ("nonfinite_q", ValueError, "q contains NaN or Inf"),
        ("nonfinite_a", ValueError, "A contains NaN or Inf"),
        ("nan_bound", ValueError, "must not contain NaN"),
        ("inconsistent_bounds", ValueError, "lower bound"),
        ("dtype_mismatch", ValueError, "must use"),
    ],
)
def test_pinned_validate_data_guards_precede_setup_and_preserve_workspace(
    case, error_type, match, monkeypatch
):
    """Translate auxil.c::validate_data without mutating committed Torch state."""
    dtype = torch.float64
    P = torch.eye(2, dtype=dtype)
    q = torch.zeros(2, dtype=dtype)
    A = torch.eye(2, dtype=dtype)
    l, u = -torch.ones(2, dtype=dtype), torch.ones(2, dtype=dtype)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": 0,
        "adaptive_rho": False,
        "polishing": False,
        "check_termination": 1,
        "return_info": True,
    }
    workspace = TorchOSQPWorkspace()
    torch_osqp_module.solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)
    workspace_fields = workspace.__dict__.copy()
    solver_fields = workspace.linear_solver.__dict__.copy()

    if case == "missing_p":
        P = None
    elif case == "missing_q":
        q = None
    elif case == "missing_a":
        A = None
    elif case == "zero_variables":
        P, q = torch.empty((0, 0), dtype=dtype), torch.empty(0, dtype=dtype)
        A, l, u = (
            torch.empty((0, 0), dtype=dtype),
            torch.empty(0, dtype=dtype),
            torch.empty(0, dtype=dtype),
        )
    elif case == "p_shape":
        P = torch.eye(3, dtype=dtype)
    elif case == "a_shape":
        A = torch.ones((2, 3), dtype=dtype)
    elif case == "bound_length":
        u = u[:1]
    elif case == "lower_triangle":
        P = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=dtype)
    elif case == "nonfinite_p":
        P = P.clone()
        P[0, 0] = torch.inf
    elif case == "nonfinite_q":
        q = q.clone()
        q[0] = torch.nan
    elif case == "nonfinite_a":
        A = A.clone()
        A[0, 0] = -torch.inf
    elif case == "nan_bound":
        l = l.clone()
        l[0] = torch.nan
    elif case == "inconsistent_bounds":
        l = u + 1.0
    elif case == "dtype_mismatch":
        P = P.to(torch.float32)

    def setup_must_not_run(*args, **kwargs):
        pytest.fail("validation must finish before workspace preparation")

    monkeypatch.setattr(torch_osqp_module, "_prepare_workspace", setup_must_not_run)
    with pytest.raises(error_type, match=match):
        torch_osqp_module.solve_torch_osqp_direct(
            P, q, A, l, u, settings, workspace
        )

    for name, value in workspace_fields.items():
        if name == "builtin_stats":
            assert workspace.builtin_stats == value
        else:
            assert workspace.__dict__[name] is value
    for name, value in solver_fields.items():
        assert workspace.linear_solver.__dict__[name] is value


def test_near_symmetric_p_is_safely_symmetrized():
    problem = list(_bound_problem())
    perturbation = 10.0 * torch.finfo(torch.float64).eps
    problem[0] = torch.tensor(
        [[2.0, 0.5 + perturbation], [0.5, 2.0]], dtype=torch.float64
    )
    expected = 0.5 * (problem[0] + problem[0].T)
    validated, *_ = torch_osqp_module._validate_qp(
        problem[0],
        problem[1].reshape(-1),
        problem[0].new_empty((0, 2)),
        problem[0].new_empty(0),
        problem[0].new_empty(0),
        {},
    )
    assert torch.equal(validated, expected)
    solution, info = _solve(tuple(problem))
    assert torch.all(torch.isfinite(solution))
    assert info["status_compatible"]


def test_upper_triangular_p_matches_pinned_osqp_storage_semantics(
    floating_dtype, available_device
):
    """Pinned OSQP expands valid upper-triangular P storage symmetrically."""
    upper_problem = list(_bound_problem(floating_dtype, available_device))
    upper_problem[0] = torch.tensor(
        [[2.0, 1.0], [0.0, 2.0]],
        device=available_device,
        dtype=floating_dtype,
    )
    full_problem = list(upper_problem)
    full_problem[0] = torch.tensor(
        [[2.0, 1.0], [1.0, 2.0]],
        device=available_device,
        dtype=floating_dtype,
    )
    settings = {"polishing": False, "adaptive_rho": False}

    upper_solution, upper_info = _solve(
        tuple(upper_problem), settings=settings
    )
    full_solution, full_info = _solve(tuple(full_problem), settings=settings)

    assert upper_info["status"] == full_info["status"] == "solved"
    tolerance = 3e-4 if floating_dtype == torch.float32 else 1e-8
    assert torch.allclose(
        upper_solution, full_solution, atol=tolerance, rtol=tolerance
    )
    assert upper_info["objective"] == pytest.approx(
        full_info["objective"], abs=tolerance, rel=tolerance
    )


def test_optional_convexity_diagnostic_rejects_indefinite_p():
    problem = list(_bound_problem())
    # Positive diagonal leaves this optional spectral check independently tested.
    problem[0] = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="positive semidefinite"):
        _solve(tuple(problem), settings={"check_convexity": True})


@pytest.mark.parametrize(
    "p,a,sigma,scaling,equality,accepted",
    [
        (-1.0, 1.0, 1e-6, 0, False, False),
        (-1.0, 1.0, 1e-6, 10, False, False),
        (-1.0, 1.0, 5.0, 0, False, True),
        (-1.0, 10.0, 1e-6, 0, False, True),
        (-1.0, 1.0, 1e-6, 0, True, True),
        (-1e8, 1.0, 5.0, 0, False, False),
        (-1e8, 1.0, 5.0, 10, False, True),
        (-2e-6, None, 1e-6, 0, False, False),
        (-5e-7, None, 1e-6, 0, False, True),
        (0.0, None, 1e-6, 0, False, True),
    ],
)
def test_setup_inertia_uses_scaled_kkt_not_unshifted_p(
    p, a, sigma, scaling, equality, accepted, floating_dtype, available_device,
    monkeypatch,
):
    # Isolate the unchanged C KKT-sign rule below the approved input safeguard.
    # These nonconvex fixtures are not accepted through the public Torch path.
    monkeypatch.setattr(torch_osqp_module, "_validate_qp", lambda *args: args[:5])
    P = torch.tensor([[p]], dtype=floating_dtype, device=available_device)
    q = P.new_zeros(1)
    A = P.new_empty((0, 1)) if a is None else P.new_full((1, 1), a)
    l = P.new_full((A.shape[0],), 0.0 if equality else -1.0)
    u = -l
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "sigma": sigma,
        "scaling": scaling,
        "adaptive_rho": False,
        "polishing": False,
        "check_termination": 1,
        "max_iter": 25,
        "eps_abs": 1e-8,
        "eps_rel": 1e-8,
        "check_convexity": False,
        "return_info": True,
    }
    workspace = TorchOSQPWorkspace()

    if not accepted:
        with pytest.raises(TorchLinearSolveError, match="KKT inertia"):
            torch_osqp_module.solve_torch_osqp_direct(
                P, q, A, l, u, settings, workspace
            )
        assert workspace.last_info is None
        assert workspace.state is None
        assert not workspace.linear_solver.ready
        assert workspace.linear_solver.solve_count == 0
    else:
        solution, info = torch_osqp_module.solve_torch_osqp_direct(
            P, q, A, l, u, settings, workspace
        )
        assert info["status"] == "solved"
        assert torch.equal(solution.reshape(-1), q)
        assert solution.device == P.device and solution.dtype == P.dtype
        assert info["primal_infeasibility_certificate"] is None
        assert info["dual_infeasibility_certificate"] is None


def test_psd_variable_block_preserves_ill_scaled_rank_deficient_kkt(
    floating_dtype, available_device, monkeypatch
):
    P = torch.zeros((2, 2), dtype=floating_dtype, device=available_device)
    q = P.new_zeros(2)
    A = P.new_full((2, 2), 1e8)
    l, u = -P.new_ones(2), P.new_ones(2)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": 0,
        "adaptive_rho": False,
        "polishing": False,
        "check_termination": 1,
        "return_info": True,
    }
    calls = []
    original = torch.linalg.eigvalsh

    def record_eigenvalues(matrix, *args, **kwargs):
        calls.append((matrix.shape, matrix.device, matrix.dtype))
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigvalsh", record_eigenvalues)
    if floating_dtype == torch.float32:
        with pytest.raises(TorchLinearSolveError, match="zeros in the diagonal"):
            torch_osqp_module.solve_torch_osqp_direct(P, q, A, l, u, settings)
        return
    solution, info = torch_osqp_module.solve_torch_osqp_direct(
        P, q, A, l, u, settings
    )
    assert info["status"] == "solved"
    assert torch.equal(solution.reshape(-1), q)
    # The positive variable block proves the signs without an ill-scaled
    # full-KKT eigensolve that can lose the small positive eigenvalue.
    assert calls == []


def test_inertia_check_failure_preserves_explicit_torch_workspace(monkeypatch):
    problem = _bound_problem()
    settings = {"polishing": False, "adaptive_rho": False, "scaling": 0}
    workspace = TorchOSQPWorkspace()
    _solve(problem, settings=settings, workspace=workspace)
    state_before, info_before = workspace.state, workspace.last_info
    matrix_before = workspace.linear_solver.matrix
    counts_before = (
        workspace.linear_solver.factorization_count,
        workspace.linear_solver.solve_count,
    )

    def fail_check(*args, **kwargs):
        raise TorchLinearSolveError("synthetic KKT inertia failure")

    monkeypatch.setattr(workspace.linear_solver, "factorize", fail_check)
    changed = list(problem)
    changed[0] = 2.0 * changed[0]
    with pytest.raises(RuntimeError, match="inertia"):
        _solve(tuple(changed), settings=settings, workspace=workspace)
    assert workspace.state is state_before
    assert workspace.last_info is info_before
    assert workspace.linear_solver.matrix is matrix_before
    assert (
        workspace.linear_solver.factorization_count,
        workspace.linear_solver.solve_count,
    ) == counts_before


@pytest.mark.parametrize("scaling", [0, 10])
@pytest.mark.parametrize("magnitude", [1e-6, 1e-20, "smallest"])
def test_sigma_shifted_negative_diagonal_reaches_kkt_setup(
    magnitude, scaling, floating_dtype, available_device, monkeypatch,
):
    P = torch.eye(2, dtype=floating_dtype, device=available_device)
    # Pinned validation permits this input; sigma makes the regularized block
    # positive. Translation must not add a stronger convexity policy.
    P[1, 1] = (
        torch.nextafter(P.new_zeros(()), P.new_tensor(-torch.inf))
        if magnitude == "smallest" else -magnitude
    )
    original = P.clone()
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": scaling,
        "sigma": 1.0,
        "adaptive_rho": False,
        "polishing": False,
        "check_convexity": False,
        "check_termination": 1,
        "return_info": True,
    }
    workspace = TorchOSQPWorkspace()

    solution, info = torch_osqp_module.solve_torch_osqp_direct(
        P, P.new_zeros(2), P.new_ones((1, 2)),
        -P.new_ones(1), P.new_ones(1), settings, workspace
    )
    assert info["status"] == "solved"
    assert torch.equal(solution.reshape(-1), P.new_zeros(2))
    assert torch.equal(P, original)
    assert workspace.linear_solver.factorization_count == 1
    assert workspace.linear_solver.solve_count > 0
    assert workspace.state is not None


@pytest.mark.parametrize("kind", ["zero", "signed_zero", "rank_one", "tiny_positive"])
def test_psd_diagonal_boundaries_reach_regularized_kkt(
    kind, floating_dtype, available_device,
):
    P = torch.zeros((3, 3), dtype=floating_dtype, device=available_device)
    if kind == "signed_zero":
        P.fill_(-0.0)
    elif kind == "rank_one":
        vector = P.new_tensor([1.0, -2.0, 0.0])
        P = torch.outer(vector, vector)
    elif kind == "tiny_positive":
        P[0, 0] = torch.nextafter(P.new_zeros(()), P.new_ones(()))
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": 0,
        "polishing": False,
        "adaptive_rho": False,
        "check_termination": 1,
        "return_info": True,
    }
    original = P.clone()
    solution, info = torch_osqp_module.solve_torch_osqp_direct(
        P, P.new_zeros(3), P.new_empty((0, 3)),
        P.new_empty(0), P.new_empty(0), settings
    )
    assert info["status"] == "solved"
    assert solution.device == P.device and solution.dtype == P.dtype
    assert torch.equal(solution.reshape(-1), P.new_zeros(3))
    assert torch.equal(P, original)


@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("sigma", [1e-6, 1.0])
def test_qdldl_zero_variable_block_pinned_setup_boundary(rows, sigma):
    # The pinned AMD ordering exposes a zero pivot with one constraint row.
    P = torch.tensor([[-sigma]], dtype=torch.float64)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": 0,
        "sigma": sigma,
        "adaptive_rho": False,
        "polishing": False,
        "return_info": True,
    }
    args = (P, P.new_zeros(1), P.new_ones((rows, 1)),
            -P.new_ones(rows), P.new_ones(rows), settings)
    if rows == 1:
        with pytest.raises(TorchLinearSolveError, match="zeros in the diagonal"):
            torch_osqp_module.solve_torch_osqp_direct(*args)
        return
    _, info = torch_osqp_module.solve_torch_osqp_direct(*args)
    assert info["status"] == "solved"


def test_requested_polishing_rejection_preserves_admm_solution(monkeypatch):
    def rejected(P, q, A, l, u, x, z, y, settings, **kwargs):
        del kwargs
        return (
            x,
            z,
            y,
            {
                "polishing": True,
                "polishing_success": False,
                "polishing_status": "rejected_no_improvement",
                "polishing_time_ms": 0.0,
                "polishing_active_constraints": 1,
                "polishing_refine_iter": 3,
            },
        )

    monkeypatch.setattr(torch_osqp_module, "_polish_solution", rejected)
    solution, info = _solve(_bound_problem(), settings={"polishing": True})
    assert torch.all(torch.isfinite(solution))
    assert info["status"] == "solved"
    assert info["polishing_status"] == "rejected_no_improvement"
    assert not info["polishing_success"]
