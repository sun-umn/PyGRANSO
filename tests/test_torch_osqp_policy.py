import warnings

import numpy as np
import pytest
import torch

from pygranso.private import osqpTorchAdapter as adapter
from pygranso.private import torchOSQP as torch_osqp_module
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchLinearSolve import TorchLinearSolveError


def _problem(n=2, device="cpu"):
    device = torch.device(device)
    dtype = torch.float64
    return (
        torch.eye(n, device=device, dtype=dtype),
        -torch.ones((n, 1), device=device, dtype=dtype),
        None,
        None,
        torch.zeros((n, 1), device=device, dtype=dtype),
        torch.ones((n, 1), device=device, dtype=dtype),
    )


def test_torch_status_codes_match_the_complete_pinned_osqp_enum():
    assert torch_osqp_module._STATUS_VALUES == {
        "solved": 1,
        "solved_inaccurate": 2,
        "primal_infeasible": 3,
        "primal_infeasible_inaccurate": 4,
        "dual_infeasible": 5,
        "dual_infeasible_inaccurate": 6,
        "max_iter_reached": 7,
        "time_limit_reached": 8,
        "non_convex": 9,
        "interrupted": 10,
        "unsolved": 11,
    }
    assert torch_osqp_module._COMPATIBLE_STATUSES == {
        "solved",
        "solved_inaccurate",
    }
    assert torch_osqp_module._SOLUTION_STATUSES == {
        "solved",
        "solved_inaccurate",
        "max_iter_reached",
        "time_limit_reached",
    }


def test_auto_cpu_follows_requested_device_and_uses_builtin():
    H, f, A, b, lower, upper = _problem()
    _, info = adapter.solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        lower,
        upper,
        "cpu",
        True,
        {"algebra": "auto", "settings": {"return_info": True}},
        TorchOSQPWorkspace(),
    )
    assert info["backend"] == "builtin"
    assert info["selection_reason"] == "cpu_target_uses_builtin"


def test_builtin_workspace_reuses_structure_for_vector_update():
    workspace = TorchOSQPWorkspace()
    problem = _problem()
    _, first = adapter.solve_osqp_torch_qp(
        *problem,
        "cpu",
        True,
        {"algebra": "builtin", "settings": {"return_info": True}},
        workspace,
    )
    changed = list(problem)
    changed[1] = -2.0 * torch.ones_like(changed[1])
    _, second = adapter.solve_osqp_torch_qp(
        *changed,
        "cpu",
        True,
        {"algebra": "builtin", "settings": {"return_info": True}},
        workspace,
    )
    assert first["builtin_workspace"]["setups"] == 1
    assert second["builtin_workspace"]["updates"] == 1
    assert second["builtin_workspace"]["last_cache_hit"]


def test_builtin_workspace_rebuilds_when_setup_settings_change():
    workspace = TorchOSQPWorkspace()
    problem = _problem()
    adapter.solve_osqp_torch_qp(
        *problem,
        "cpu",
        True,
        {
            "algebra": "builtin",
            "settings": {"return_info": True, "polishing": False},
        },
        workspace,
    )
    _, info = adapter.solve_osqp_torch_qp(
        *problem,
        "cpu",
        True,
        {
            "algebra": "builtin",
            "settings": {
                "return_info": True,
                "polishing": False,
                "max_iter": 1,
            },
        },
        workspace,
    )
    assert info["status"] == "max_iter_reached"
    assert not info["builtin_workspace"]["last_cache_hit"]
    assert info["builtin_workspace"]["setups"] == 2
    assert info["builtin_workspace"]["rebuilds"] == 1
    assert workspace.builtin_cache is None


def test_builtin_cache_respects_disabled_warm_start(monkeypatch):
    workspace = TorchOSQPWorkspace()
    H, f, A, b, lower, upper = _problem()
    options = {
        "algebra": "builtin",
        "settings": {
            "return_info": True,
            "polishing": False,
            "warm_start": False,
        },
    }
    adapter.solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        lower,
        upper,
        "cpu",
        True,
        options,
        workspace,
    )

    def forbidden_warm_start(*args, **kwargs):
        pytest.fail("warm_start=False must suppress cached explicit warm starts")

    monkeypatch.setattr(
        workspace.builtin_cache["problem"], "warm_start", forbidden_warm_start
    )
    changed_f = f - 0.25
    _, info = adapter.solve_osqp_torch_qp(
        H,
        changed_f,
        A,
        b,
        lower,
        upper,
        "cpu",
        True,
        options,
        workspace,
    )
    assert info["builtin_workspace"]["last_cache_hit"]


def test_builtin_cache_warm_starts_from_raw_official_solution(monkeypatch):
    raw_x = np.array([0.5, 0.5])
    raw_y = np.array([0.1, 0.2])
    warm_starts = []
    solve_options = []

    class FakeProblem:
        def setup(self, *args, **kwargs):
            pass

        def update(self, **kwargs):
            pass

        def warm_start(self, *, x, y):
            warm_starts.append((np.array(x, copy=True), np.array(y, copy=True)))

        def solve(self, **kwargs):
            solve_options.append(kwargs)
            info = type(
                "Info",
                (),
                {
                    "status": "solved",
                    "status_val": 1,
                    "status_polish": -1,
                    "obj_val": -0.5,
                    "prim_res": 0.0,
                    "dual_res": 0.0,
                    "iter": 25,
                },
            )()
            return type(
                "Result",
                (),
                {
                    "info": info,
                    "x": raw_x.copy(),
                    "y": raw_y.copy(),
                    "prim_inf_cert": None,
                    "dual_inf_cert": None,
                },
            )()

    fake_osqp = type("FakeOSQP", (), {"OSQP": lambda self, algebra: FakeProblem()})()
    monkeypatch.setattr(adapter, "_import_osqp", lambda: fake_osqp)
    monkeypatch.setattr(
        torch_osqp_module,
        "_polish_solution",
        lambda *args, **kwargs: pytest.fail(
            "builtin adapter must not post-process the official OSQP result"
        ),
    )
    workspace = TorchOSQPWorkspace()
    options = {
        "algebra": "builtin",
        "settings": {"return_info": True, "polishing": True},
    }

    adapter.solve_osqp_torch_qp(*_problem(), "cpu", True, options, workspace)
    _, info = adapter.solve_osqp_torch_qp(
        *_problem(), "cpu", True, options, workspace
    )

    assert info["builtin_workspace"]["last_cache_hit"]
    assert solve_options == [{"raise_error": False}, {"raise_error": False}]
    assert len(warm_starts) == 1
    assert np.array_equal(warm_starts[0][0], raw_x)
    assert np.array_equal(warm_starts[0][1], raw_y)


def test_builtin_validation_failure_preserves_prior_cache_and_stats():
    workspace = TorchOSQPWorkspace()
    problem = _problem()
    adapter.solve_osqp_torch_qp(
        *problem,
        "cpu",
        True,
        {
            "algebra": "builtin",
            "settings": {"return_info": True, "polishing": False},
        },
        workspace,
    )
    prior_cache = workspace.builtin_cache
    prior_stats = dict(workspace.builtin_stats)
    invalid = list(problem)
    invalid[0] = invalid[0].clone()
    invalid[0][0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        adapter.solve_osqp_torch_qp(
            *invalid,
            "cpu",
            True,
            {
                "algebra": "builtin",
                "settings": {"return_info": True, "polishing": False},
            },
            workspace,
        )
    assert workspace.builtin_cache is prior_cache
    assert workspace.builtin_stats == prior_stats


def test_builtin_cached_mutation_failure_clears_cache_transactionally(monkeypatch):
    workspace = TorchOSQPWorkspace()
    H, f, A, b, lower, upper = _problem()
    options = {
        "algebra": "builtin",
        "settings": {"return_info": True, "polishing": False},
    }
    adapter.solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        lower,
        upper,
        "cpu",
        True,
        options,
        workspace,
    )
    prior_info = workspace.last_info

    def failed_update(*args, **kwargs):
        raise RuntimeError("injected cached update failure")

    monkeypatch.setattr(workspace.builtin_cache["problem"], "update", failed_update)
    with pytest.raises(RuntimeError, match="injected cached update failure"):
        adapter.solve_osqp_torch_qp(
            H,
            f - 0.5,
            A,
            b,
            lower,
            upper,
            "cpu",
            True,
            options,
            workspace,
        )
    assert workspace.builtin_cache is None
    assert workspace.last_info is prior_info


def test_builtin_accepts_nan_iterates_only_with_matching_infeasibility_certificate(
    monkeypatch,
):
    class FakeProblem:
        def setup(self, *args, **kwargs):
            pass

        def solve(self, **kwargs):
            info = type(
                "Info",
                (),
                {
                    "status": "primal infeasible",
                    "status_val": 3,
                    "status_polish": 0,
                    "obj_val": float("inf"),
                    "prim_res": 1.0,
                    "dual_res": 0.0,
                    "iter": 25,
                },
            )()
            return type(
                "Result",
                (),
                {
                    "info": info,
                    "x": np.full(2, np.nan),
                    "y": np.full(2, np.nan),
                    "prim_inf_cert": np.array([1.0, -1.0]),
                    "dual_inf_cert": np.full(2, np.nan),
                },
            )()

    fake_osqp = type("FakeOSQP", (), {"OSQP": lambda self, algebra: FakeProblem()})()
    monkeypatch.setattr(adapter, "_import_osqp", lambda: fake_osqp)
    solution, info = adapter.solve_osqp_torch_qp(
        *_problem(),
        "cpu",
        True,
        {
            "algebra": "builtin",
            "settings": {"return_info": True, "polishing": False},
        },
        TorchOSQPWorkspace(),
    )
    assert info["status"] == "primal_infeasible"
    assert torch.all(torch.isnan(solution))
    assert info["primal_infeasibility_certificate"] is not None
    assert info["dual_infeasibility_certificate"] is None


def test_builtin_rejects_infeasibility_status_without_matching_certificate(
    monkeypatch,
):
    class FakeProblem:
        def setup(self, *args, **kwargs):
            pass

        def solve(self, **kwargs):
            info = type(
                "Info",
                (),
                {"status": "dual infeasible", "status_val": 5},
            )()
            return type(
                "Result",
                (),
                {
                    "info": info,
                    "x": np.full(2, np.nan),
                    "y": np.full(2, np.nan),
                    "prim_inf_cert": None,
                    "dual_inf_cert": None,
                },
            )()

    fake_osqp = type("FakeOSQP", (), {"OSQP": lambda self, algebra: FakeProblem()})()
    monkeypatch.setattr(adapter, "_import_osqp", lambda: fake_osqp)
    with pytest.raises(RuntimeError, match="without a valid certificate"):
        adapter.solve_osqp_torch_qp(
            *_problem(),
            "cpu",
            True,
            {
                "algebra": "builtin",
                "settings": {"return_info": True, "polishing": False},
            },
            TorchOSQPWorkspace(),
        )


@pytest.mark.parametrize("algebra", ["builtin", "auto"])
def test_builtin_interrupted_status_is_cancellation_not_a_finite_solution(
    monkeypatch, algebra,
):
    class FakeProblem:
        def setup(self, *args, **kwargs):
            pass

        def solve(self, **kwargs):
            info = type(
                "Info",
                (),
                {
                    "status": "interrupted by user",
                    "status_val": 10,
                    "status_polish": 0,
                    "obj_val": -0.5,
                    "prim_res": 1.0,
                    "dual_res": 1.0,
                    "iter": 1,
                },
            )()
            return type(
                "Result",
                (),
                {
                    "info": info,
                    "x": np.array([0.5, 0.5]),
                    "y": np.zeros(2),
                    "prim_inf_cert": None,
                    "dual_inf_cert": None,
                },
            )()

    fake_osqp = type("FakeOSQP", (), {"OSQP": lambda self, algebra: FakeProblem()})()
    monkeypatch.setattr(adapter, "_import_osqp", lambda: fake_osqp)
    workspace = TorchOSQPWorkspace()
    workspace.active_backend = "torch"
    workspace.state = {"x": torch.tensor([0.25, 0.75], dtype=torch.float64)}
    workspace.last_info = {"status": "solved", "backend": "torch"}
    prior_state = {key: value.clone() for key, value in workspace.state.items()}

    with pytest.raises(KeyboardInterrupt, match="Builtin OSQP.*interrupted"):
        adapter.solve_osqp_torch_qp(
            *_problem(),
            "cpu",
            True,
            {
                "algebra": algebra,
                "settings": {"return_info": True, "polishing": False},
            },
            workspace,
        )

    assert workspace.active_backend == "torch"
    assert set(workspace.state) == set(prior_state)
    for key, value in prior_state.items():
        assert torch.equal(workspace.state[key], value)
    assert workspace.builtin_cache is None
    assert workspace.last_info["status"] == "interrupted"
    assert workspace.last_info["status_val"] == 10
    assert not workspace.last_info["status_compatible"]
    assert workspace.last_info["requested_backend"] == algebra
    assert workspace.last_info["fallback"] == {"occurred": False}
    assert workspace.last_info["primal_infeasibility_certificate"] is None
    assert workspace.last_info["dual_infeasibility_certificate"] is None


def test_builtin_interrupted_cached_update_discards_mutated_problem(monkeypatch):
    class FakeProblem:
        def __init__(self):
            self.solve_calls = 0

        def setup(self, *args, **kwargs):
            pass

        def update(self, **kwargs):
            pass

        def warm_start(self, **kwargs):
            pass

        def solve(self, **kwargs):
            self.solve_calls += 1
            interrupted = self.solve_calls == 2
            info = type(
                "Info",
                (),
                {
                    "status": "interrupted by user" if interrupted else "solved",
                    "status_val": 10 if interrupted else 1,
                    "status_polish": 0,
                    "obj_val": -1.0,
                    "prim_res": 0.0 if not interrupted else 1.0,
                    "dual_res": 0.0 if not interrupted else 1.0,
                    "iter": 1,
                },
            )()
            return type(
                "Result",
                (),
                {
                    "info": info,
                    "x": np.ones(2),
                    "y": np.zeros(2),
                    "prim_inf_cert": None,
                    "dual_inf_cert": None,
                },
            )()

    problem = FakeProblem()
    fake_osqp = type("FakeOSQP", (), {"OSQP": lambda self, algebra: problem})()
    monkeypatch.setattr(adapter, "_import_osqp", lambda: fake_osqp)
    workspace = TorchOSQPWorkspace()
    options = {
        "algebra": "builtin",
        "settings": {"return_info": True, "polishing": False},
    }
    H, f, A, b, lower, upper = _problem()
    adapter.solve_osqp_torch_qp(
        H, f, A, b, lower, upper, "cpu", True, options, workspace
    )
    prior_stats = dict(workspace.builtin_stats)

    with pytest.raises(KeyboardInterrupt, match="Builtin OSQP.*interrupted"):
        adapter.solve_osqp_torch_qp(
            H,
            f - 0.5,
            A,
            b,
            lower,
            upper,
            "cpu",
            True,
            options,
            workspace,
        )

    assert workspace.active_backend == "builtin"
    assert workspace.builtin_cache is None
    assert workspace.builtin_stats == prior_stats
    assert workspace.last_info["status"] == "interrupted"
    assert workspace.last_info["status_val"] == 10


def test_builtin_fallback_interruption_preserves_first_torch_diagnostics(monkeypatch):
    def select_torch(*args, **kwargs):
        return {
            "backend": "torch",
            "requested_backend": "auto",
            "selection_reason": "injected_torch_first",
            "estimated_kkt_dim": 4,
            "estimated_dense_working_memory_mb": 0.0,
            "estimated_dense_memory_components_mb": {},
        }

    def unsolved_torch(*args, **kwargs):
        return torch.zeros((2, 1), dtype=torch.float64), {
            "status": "max_iter_reached",
            "status_val": 7,
            "status_compatible": False,
        }

    def interrupted_builtin(*args, **kwargs):
        return torch.full((2, 1), 0.5, dtype=torch.float64), {
            "status": "interrupted",
            "status_val": 10,
            "status_compatible": False,
            "backend": "builtin",
            "fallback": {"occurred": False},
            "primal_infeasibility_certificate": None,
            "dual_infeasibility_certificate": None,
        }

    monkeypatch.setattr(adapter, "_select_backend", select_torch)
    monkeypatch.setattr(adapter, "_solve_torch_osqp_path", unsolved_torch)
    monkeypatch.setattr(adapter, "_solve_builtin_osqp_path", interrupted_builtin)
    workspace = TorchOSQPWorkspace()
    workspace.active_backend = "torch"
    workspace.state = {"x": torch.tensor([0.25, 0.75], dtype=torch.float64)}
    prior_state = {key: value.clone() for key, value in workspace.state.items()}

    with pytest.warns(RuntimeWarning, match="falling back"):
        with pytest.raises(KeyboardInterrupt, match="Builtin OSQP.*interrupted"):
            adapter.solve_osqp_torch_qp(
                *_problem(),
                "cpu",
                True,
                {"algebra": "auto", "settings": {"return_info": True}},
                workspace,
            )

    assert workspace.active_backend == "torch"
    assert set(workspace.state) == set(prior_state)
    for key, value in prior_state.items():
        assert torch.equal(workspace.state[key], value)
    assert workspace.last_info["status"] == "interrupted"
    assert workspace.last_info["status_val"] == 10
    fallback = workspace.last_info["fallback"]
    assert fallback["occurred"]
    assert fallback["trigger"] == "unsolved_status"
    assert fallback["status"] == "max_iter_reached"
    assert fallback["torch_info"]["status"] == "max_iter_reached"


@pytest.mark.parametrize(
    "legacy_setting",
    [
        "linear_solver",
        "cg_rtol",
        "cg_atol",
        "cg_max_iter",
        "cg_check_interval",
        "cg_fixed_iters",
        "linear_solver_auto_sparse_min_kkt_dim",
        "linear_solver_auto_min_kkt_dim",
        "linear_solver_auto_max_density",
        "linear_solver_auto_dense_memory_limit_mb",
        "cuda_event_timing",
        "torch_compile_admm",
        "cuda_graph",
    ],
)
def test_legacy_solver_settings_warn_and_raise(monkeypatch, legacy_setting):
    H, f, A, b, lower, upper = _problem()

    def unexpected(*args, **kwargs):
        raise AssertionError("validation must precede backend/workspace creation")

    monkeypatch.setattr(adapter, "TorchOSQPWorkspace", unexpected)
    monkeypatch.setattr(adapter, "_select_backend", unexpected)
    with pytest.warns(FutureWarning, match="archived"):
        with pytest.raises(ValueError, match=rf"{legacy_setting}.*Remove these keys"):
            adapter.solve_osqp_torch_qp(
                H,
                f,
                A,
                b,
                lower,
                upper,
                "cpu",
                True,
                {
                    "algebra": "torch",
                    "settings": {legacy_setting: "archived"},
                },
            )


def test_settings_match_supported_boundaries_and_defaults():
    settings = adapter._normalize_options(
        {
            "settings": {
                "check_termination": 0,
                "rho_update_interval": 0,
                "rho_update_tolerance": 1.0,
            }
        },
        torch.float64,
    )["settings"]
    assert settings["check_termination"] == 0
    assert settings["rho_update_interval"] == 0
    assert settings["rho_update_tolerance"] == 1.0
    assert settings["adaptive_rho_fraction"] == 0.4
    assert settings["eps_prim_inf"] == 1e-4
    assert settings["eps_dual_inf"] == 1e-4
    assert settings["time_limit"] == 1e10
    assert settings["polishing"] is False
    builtin = adapter._builtin_settings(settings)
    assert builtin["check_termination"] == 0
    assert builtin["time_limit"] == settings["time_limit"]
    assert builtin["adaptive_rho_fraction"] == 0.4
    assert builtin["delta"] == settings["polish_delta"]
    assert builtin["polish_refine_iter"] == settings["polish_refine_iter"]
    assert builtin["polishing"] is False
    direct_default = torch_osqp_module._default_polish_info({})
    assert direct_default["polishing"] is False
    assert direct_default["polishing_status"] == "disabled"


@pytest.mark.parametrize(
    "settings,match",
    [
        ({"rho": 0.0}, "positive"),
        ({"rho": -0.1}, "positive"),
        ({"rho": float("nan")}, "finite"),
        ({"sigma": -0.1}, "positive"),
        ({"alpha": 2.0}, r"in \(0, 2\)"),
        pytest.param(
            {"max_iter": -1}, "positive integer", id="update-max-iter-minus-one"
        ),
        ({"eps_abs": -1.0}, "nonnegative"),
        pytest.param({"eps_abs": -1.1}, "nonnegative", id="setup-eps-abs"),
        pytest.param({"eps_rel": -1.0}, "nonnegative", id="update-eps-rel"),
        ({"eps_rel": -0.1}, "nonnegative"),
        ({"eps_abs": float("inf")}, "finite"),
        ({"eps_abs": 0.0, "eps_rel": 0.0}, "At least one"),
        pytest.param(
            {"eps_prim_inf": -0.1}, "positive", id="eps-prim-inf-minus-point-one"
        ),
        ({"eps_prim_inf": 0.0}, "positive"),
        pytest.param(
            {"eps_dual_inf": -0.1}, "positive", id="update-eps-dual-inf"
        ),
        pytest.param({"eps_dual_inf": 0.0}, "positive", id="setup-eps-dual-inf"),
        ({"eps_dual_inf": -1.0}, "positive"),
        ({"max_iter": 0}, "positive integer"),
        ({"time_limit": 0.0}, "positive"),
        ({"time_limit": -0.2}, "positive"),
        ({"time_limit": float("nan")}, "finite"),
        ({"time_limit": float("inf")}, "finite"),
        pytest.param(
            {"rho_update_interval": -1},
            "nonnegative integer",
            id="adaptive-rho-interval-minus-one",
        ),
        ({"polish_refine_iter": -1}, "nonnegative integer"),
        pytest.param(
            {"polish_refine_iter": -3},
            "nonnegative integer",
            id="setup-polish-refine-iter",
        ),
        ({"polish_delta": 0.0}, "positive"),
        pytest.param({"polish_delta": -1.1}, "positive", id="setup-polish-delta"),
        ({"scaling": -1}, "nonnegative integer"),
        ({"symmetry_tolerance_multiplier": 0.0}, "positive"),
        ({"check_linear_residual": 1}, "must be a bool"),
        *[({"check_dualgap": value}, "must be a bool") for value in
          (0, 1, -1, 2, "false", None, float("nan"), float("inf"), torch.tensor(True))],
        *[({"scaled_termination": value}, "must be a bool") for value in
          (0, 1, -1, 2, "false", None, float("nan"), float("inf"), torch.tensor(True))],
        ({"check_termination": -1}, "nonnegative"),
        pytest.param(
            {"rho_update_tolerance": 0.5},
            "at least 1.0",
            id="adaptive-rho-tolerance-point-five",
        ),
        ({"rho_update_tolerance": 0.999}, "at least 1.0"),
        ({"adaptive_rho": 4}, r"mode in \{0,1,2,3\}"),
        pytest.param(
            {"adaptive_rho": 9}, r"mode in \{0,1,2,3\}", id="adaptive-rho-enum-nine"
        ),
        ({"adaptive_rho": -1}, r"mode in \{0,1,2,3\}"),
        ({"adaptive_rho": 1.0}, r"mode in \{0,1,2,3\}"),
        ({"adaptive_rho": "1"}, r"mode in \{0,1,2,3\}"),
        ({"adaptive_rho_fraction": 0.0}, "positive"),
        ({"adaptive_rho_fraction": -0.1}, "positive"),
        ({"adaptive_rho_fraction": float("nan")}, "finite"),
        ({"polishing": "false"}, "must be a bool"),
        pytest.param({"polishing": 2}, "must be a bool", id="polishing-two"),
        ({"warm_start": "false"}, "must be a bool"),
        pytest.param({"warm_start": -1}, "must be a bool", id="warm-start-minus-one"),
        pytest.param({"warm_start": 5}, "must be a bool", id="warm-start-five"),
        ({"verbose": 1}, "must be a bool"),
        pytest.param({"verbose": 2}, "must be a bool", id="verbose-two"),
    ],
)
def test_invalid_settings_are_rejected_before_workspace_mutation(settings, match):
    workspace = TorchOSQPWorkspace()
    workspace.active_backend = "builtin"
    workspace.builtin_cache = {"sentinel": True}
    H, f, A, b, lower, upper = _problem()
    with pytest.raises(ValueError, match=match):
        adapter.solve_osqp_torch_qp(
            H,
            f,
            A,
            b,
            lower,
            upper,
            "cpu",
            True,
            {"algebra": "torch", "settings": settings},
            workspace,
        )
    assert workspace.active_backend == "builtin"
    assert workspace.builtin_cache == {"sentinel": True}


@pytest.mark.parametrize(
    "setting",
    [
        "rho",
        "sigma",
        "alpha",
        "max_iter",
        "time_limit",
        "eps_abs",
        "eps_rel",
        "eps_prim_inf",
        "eps_dual_inf",
        "check_termination",
        "scaling",
        "rho_update_interval",
        "adaptive_rho_fraction",
        "rho_update_tolerance",
        "polish_delta",
        "polish_refine_iter",
        "symmetry_tolerance_multiplier",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="bool-true"),
        pytest.param(False, id="bool-false"),
        pytest.param(torch.tensor(1.0), id="scalar-tensor"),
        pytest.param(1.0 + 0.0j, id="complex-scalar"),
    ],
)
def test_numeric_settings_reject_ambiguous_types_before_workspace_mutation(
    setting, value,
):
    workspace = TorchOSQPWorkspace()
    workspace.active_backend = "builtin"
    workspace.builtin_cache = {"sentinel": True}
    workspace.last_info = {"status": "solved", "backend": "builtin"}
    H, f, A, b, lower, upper = _problem()

    with pytest.raises(ValueError):
        adapter.solve_osqp_torch_qp(
            H,
            f,
            A,
            b,
            lower,
            upper,
            "cpu",
            True,
            {"algebra": "torch", "settings": {setting: value}},
            workspace,
        )

    assert workspace.active_backend == "builtin"
    assert workspace.builtin_cache == {"sentinel": True}
    assert workspace.last_info == {"status": "solved", "backend": "builtin"}


@pytest.mark.parametrize(
    "setting,value",
    [
        pytest.param("alpha", np.nextafter(2.0, 0.0), id="alpha-below-two"),
        pytest.param("eps_abs", 0.0, id="eps-abs-zero-with-positive-relative"),
        pytest.param("eps_rel", 0.0, id="eps-rel-zero-with-positive-absolute"),
        pytest.param("check_termination", 0, id="termination-disabled"),
        pytest.param("scaling", 0, id="scaling-disabled"),
        pytest.param("rho_update_interval", 0, id="automatic-rho-interval"),
        pytest.param("rho_update_tolerance", 1.0, id="rho-tolerance-minimum"),
        pytest.param("time_limit", 1e-12, id="positive-time-limit"),
        pytest.param("polish_refine_iter", 0, id="polish-refinement-disabled"),
    ],
)
def test_supported_setting_boundary_is_preserved_by_normalization(setting, value):
    settings = adapter._normalize_options(
        {"algebra": "torch", "settings": {setting: value}},
        torch.float64,
    )["settings"]

    assert settings[setting] == value


@pytest.mark.parametrize(
    "requested,expected",
    [(False, False), (True, True), (0, 0), (1, 1), (2, 2), (3, 3)],
)
def test_adaptive_rho_bool_and_source_modes_are_publicly_supported(
    requested, expected,
):
    settings = adapter._normalize_options(
        {"algebra": "torch", "settings": {"adaptive_rho": requested}},
        torch.float64,
    )["settings"]
    assert settings["adaptive_rho"] == expected
    assert adapter._builtin_settings(settings)["adaptive_rho"] == expected


@pytest.mark.parametrize("mode", [2, 3])
def test_extended_rho_modes_reach_the_builtin_reference_adapter(mode):
    solution, info = adapter.solve_osqp_torch_qp(
        *_problem(),
        "cpu",
        True,
        {
            "algebra": "builtin",
            "settings": {
                "adaptive_rho": mode,
                "adaptive_rho_fraction": 0.5,
                "rho_update_interval": 1,
                "return_info": True,
                "polishing": False,
            },
        },
        TorchOSQPWorkspace(),
    )
    assert info["backend"] == "builtin"
    assert info["status"] == "solved"
    assert torch.all(torch.isfinite(solution))


@pytest.mark.parametrize(
    "setting,value",
    [
        pytest.param("linsys_solver", "unknown", id="linsys-solver"),
    ],
)
def test_unsupported_solver_features_fail_before_backend_selection(
    monkeypatch, setting, value
):
    workspace = TorchOSQPWorkspace()
    workspace.active_backend = "builtin"
    workspace.builtin_cache = {"sentinel": True}
    workspace.last_info = {"status": "solved", "backend": "builtin"}
    H, f, A, b, lower, upper = _problem()

    def unexpected(*args, **kwargs):
        raise AssertionError("unsupported settings must fail before backend selection")

    monkeypatch.setattr(adapter, "_select_backend", unexpected)
    with pytest.raises(
        ValueError, match=rf"Unsupported common OSQP setting '{setting}'"
    ):
        adapter.solve_osqp_torch_qp(
            H,
            f,
            A,
            b,
            lower,
            upper,
            "cpu",
            True,
            {
                "algebra": "torch",
                "settings": {setting: value},
            },
            workspace,
        )

    assert workspace.active_backend == "builtin"
    assert workspace.builtin_cache == {"sentinel": True}
    assert workspace.last_info == {"status": "solved", "backend": "builtin"}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_duality_gap_dtype_default_and_builtin_forwarding(dtype):
    settings = adapter._default_settings(dtype)
    assert settings["check_dualgap"] is (dtype == torch.float64)
    assert settings["scaled_termination"] is False
    for enabled in (True, False):
        options = adapter._normalize_options(
            {
                "algebra": "torch",
                "settings": {
                    "check_dualgap": enabled,
                    "scaled_termination": enabled,
                },
            },
            dtype,
        )
        assert options["settings"]["check_dualgap"] is enabled
        assert options["settings"]["scaled_termination"] is enabled
        assert adapter._builtin_settings(options["settings"])["check_dualgap"] is enabled
        assert (
            adapter._builtin_settings(options["settings"])["scaled_termination"]
            is enabled
        )


def test_invalid_torch_data_does_not_switch_or_clear_previous_backend_state():
    workspace = TorchOSQPWorkspace()
    workspace.active_backend = "builtin"
    workspace.builtin_cache = {"sentinel": True}
    workspace.last_info = {"status": "solved", "backend": "builtin"}
    H, f, A, b, lower, upper = _problem()
    H[0, 0] = torch.nan

    with pytest.raises(ValueError, match="NaN or Inf"):
        adapter.solve_osqp_torch_qp(
            H,
            f,
            A,
            b,
            lower,
            upper,
            "cpu",
            True,
            {"algebra": "torch"},
            workspace,
        )

    assert workspace.active_backend == "builtin"
    assert workspace.builtin_cache == {"sentinel": True}
    assert workspace.last_info == {"status": "solved", "backend": "builtin"}


@pytest.mark.parametrize("missing", ["A", "b"])
def test_equality_matrix_and_rhs_must_be_supplied_together(missing):
    H, f, _A, _b, lower, upper = _problem()
    A = torch.ones((1, f.numel()), dtype=f.dtype)
    b = torch.tensor(1.0, dtype=f.dtype)
    if missing == "A":
        A = None
    else:
        b = None
    with pytest.raises(ValueError, match="A and b must either both"):
        adapter.solve_osqp_torch_qp(
            H,
            f,
            A,
            b,
            lower,
            upper,
            "cpu",
            True,
            {"algebra": "torch"},
            TorchOSQPWorkspace(),
        )


@pytest.mark.parametrize("layout", ["coo", "csr", "csc"])
def test_sparse_inputs_use_visible_on_device_dense_compatibility(
    monkeypatch, available_device, layout
):
    H, f, _A, _b, lower, upper = _problem(device=available_device)
    A = torch.tensor(
        [[1.0, 1.0]], device=available_device, dtype=torch.float64
    )
    b = torch.tensor(1.0, device=available_device, dtype=torch.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse CSR tensor support is in beta state",
            category=UserWarning,
        )
        if layout == "coo":
            H = H.to_sparse()
            A = A.to_sparse()
        elif layout == "csr":
            H = H.to_sparse_csr()
            A = A.to_sparse_csr()
        else:
            H = H.to_sparse_csc()
            A = A.to_sparse_csc()

    def forbid_numpy(*args, **kwargs):
        pytest.fail("the Torch sparse compatibility path must stay in PyTorch")

    monkeypatch.setattr(adapter, "_to_numpy", forbid_numpy)
    with pytest.warns(RuntimeWarning, match="dense compatibility"):
        solution, info = adapter.solve_osqp_torch_qp(
            H,
            f,
            A,
            b,
            lower,
            upper,
            available_device,
            True,
            {
                "algebra": "torch",
                "settings": {"return_info": True, "polishing": False},
            },
            TorchOSQPWorkspace(),
        )
    diagnostics = info["sparse_to_dense"]
    assert torch.all(torch.isfinite(solution))
    assert diagnostics["occurred"]
    assert set(diagnostics["inputs"]) == {"H", "A"}
    assert diagnostics["conversion_device"] == str(H.device)
    assert diagnostics["device_transfer"] is False
    assert diagnostics["method"] == "Tensor.to(device).to_dense"


def test_explicit_lower_triangle_sparse_zero_is_a_dense_boundary_case():
    """CSC slot validation is unavailable after the declared dense conversion."""
    dtype = torch.float64
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    values = torch.tensor([2.0, 1.0, 0.0, 2.0], dtype=dtype)
    H = torch.sparse_coo_tensor(indices, values, (2, 2), check_invariants=True)
    f = -torch.ones((2, 1), dtype=dtype)
    lower = torch.zeros((2, 1), dtype=dtype)
    upper = torch.ones((2, 1), dtype=dtype)

    # Pinned C rejects the stored (1, 0) slot even though its value is zero.
    # The active Torch contract reports densification, after which that C-only
    # structural distinction no longer exists and upper numerical semantics apply.
    assert H._nnz() == 4
    with pytest.warns(RuntimeWarning, match="dense compatibility"):
        sparse_solution, sparse_info = adapter.solve_osqp_torch_qp(
            H,
            f,
            None,
            None,
            lower,
            upper,
            "cpu",
            True,
            {
                "algebra": "torch",
                "settings": {
                    "return_info": True,
                    "polishing": False,
                    "adaptive_rho": False,
                },
            },
            TorchOSQPWorkspace(),
        )
    dense_solution, dense_info = adapter.solve_osqp_torch_qp(
        torch.tensor([[2.0, 1.0], [0.0, 2.0]], dtype=dtype),
        f,
        None,
        None,
        lower,
        upper,
        "cpu",
        True,
        {
            "algebra": "torch",
            "settings": {
                "return_info": True,
                "polishing": False,
                "adaptive_rho": False,
            },
        },
        TorchOSQPWorkspace(),
    )

    assert sparse_info["status"] == dense_info["status"] == "solved"
    assert sparse_info["sparse_to_dense"]["occurred"]
    assert set(sparse_info["sparse_to_dense"]["inputs"]) == {"H"}
    assert torch.allclose(sparse_solution, dense_solution, atol=1e-8, rtol=1e-8)


def test_dense_memory_estimate_accounts_for_all_dense_compatibility_storage():
    H, f, _A, _b, *_ = _problem()
    A = torch.ones((1, 2), dtype=torch.float64)
    b = torch.tensor(1.0, dtype=torch.float64)
    selection = adapter._select_backend(
        "torch", torch.device("cpu"), torch.float64, H, f, A, b
    )
    components = selection["estimated_dense_memory_components_mb"]
    assert selection["estimated_kkt_dim"] == 5
    assert set(components) == {
        "source_matrices",
        "bound_identity",
        "canonical_constraint_matrix",
        "kkt_matrix",
        "kkt_factorization_workspace",
    }
    assert all(value > 0 for value in components.values())
    assert selection["estimated_dense_working_memory_mb"] == pytest.approx(
        sum(components.values())
    )


def test_auto_device_move_is_visible_even_for_dense_inputs():
    H, f, A, b, lower, upper = _problem()
    diagnostics = adapter._sparse_to_dense_diagnostics(
        {"H": H, "f": f, "A": A, "b": b, "LB": lower, "UB": upper},
        torch.device("cuda"),
    )
    assert not diagnostics["occurred"]
    assert diagnostics["device_transfer"]
    assert set(diagnostics["device_transfers"]) == {"H", "f", "LB", "UB"}


def test_explicit_oversize_warns_but_remains_torch_selection():
    n = 1201
    H = torch.empty((n, n), device="meta")
    f = torch.empty((n, 1), device="meta")
    with pytest.warns(RuntimeWarning, match="exceeds the validated"):
        selection = adapter._select_backend(
            "torch", torch.device("cpu"), torch.float64, H, f, None, None
        )
    assert selection["backend"] == "torch"
    assert selection["estimated_kkt_dim"] > adapter.MAX_SUPPORTED_KKT_DIM


def test_auto_failure_records_full_fallback(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to select the automatic Torch route.")
    problem = _problem(device="cuda")
    monkeypatch.setitem(adapter.PROMOTED_ACCELERATOR_BACKENDS, "cuda", True)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic torch failure")

    monkeypatch.setattr(adapter, "_solve_torch_osqp_path", fail)
    with pytest.warns(RuntimeWarning, match="synthetic torch failure"):
        _, info = adapter.solve_osqp_torch_qp(
            *problem,
            torch.device("cuda"),
            True,
            {"algebra": "auto", "settings": {"return_info": True}},
            TorchOSQPWorkspace(),
        )
    fallback = info["fallback"]
    assert fallback["occurred"]
    assert fallback["exception_type"] == "RuntimeError"
    assert fallback["fallback_backend"] == "builtin"
    assert fallback["device_transfer"]


@pytest.mark.parametrize(
    "exception_type,message",
    [
        pytest.param(ValueError, "synthetic validated-data failure", id="validation"),
        pytest.param(
            TorchLinearSolveError,
            "synthetic factorization failure",
            id="linear-system",
        ),
        pytest.param(MemoryError, "synthetic allocation failure", id="allocation"),
        pytest.param(
            torch.OutOfMemoryError,
            "synthetic device allocation failure",
            id="device-allocation",
        ),
    ],
)
def test_explicit_torch_preserves_error_family_and_committed_workspace(
    exception_type, message, monkeypatch,
):
    workspace = TorchOSQPWorkspace()
    workspace.active_backend = "torch"
    workspace.state = {"x": torch.tensor([0.25, 0.75], dtype=torch.float64)}
    workspace.last_info = {"status": "solved", "status_val": 1}
    state_before = workspace.state
    info_before = workspace.last_info

    def fail(*args, **kwargs):
        raise exception_type(message)

    def forbid_builtin(*args, **kwargs):
        pytest.fail("an explicit Torch failure must never select builtin OSQP")

    monkeypatch.setattr(adapter, "_solve_torch_osqp_path", fail)
    monkeypatch.setattr(adapter, "_solve_builtin_osqp_path", forbid_builtin)
    with pytest.raises(exception_type, match=message):
        adapter.solve_osqp_torch_qp(
            *_problem(),
            torch.device("cpu"),
            True,
            {"algebra": "torch", "settings": {"return_info": True}},
            workspace,
        )

    assert workspace.active_backend == "torch"
    assert workspace.state is state_before
    assert workspace.last_info is info_before
    assert workspace.last_info == {"status": "solved", "status_val": 1}


@pytest.mark.parametrize(
    "exception_type,message",
    [
        pytest.param(ValueError, "synthetic validated-data failure", id="validation"),
        pytest.param(
            TorchLinearSolveError,
            "synthetic factorization failure",
            id="linear-system",
        ),
        pytest.param(MemoryError, "synthetic allocation failure", id="allocation"),
        pytest.param(
            torch.OutOfMemoryError,
            "synthetic device allocation failure",
            id="device-allocation",
        ),
    ],
)
def test_auto_fallback_records_original_error_family_without_false_status(
    exception_type, message, monkeypatch,
):
    workspace = TorchOSQPWorkspace()
    monkeypatch.setattr(
        adapter,
        "_select_backend",
        lambda *args, **kwargs: {
            "requested_backend": "auto",
            "backend": "torch",
            "selection_reason": "synthetic_validated_target",
            "estimated_kkt_dim": 4,
            "estimated_dense_working_memory_mb": 1.0,
            "fallback": {"occurred": False},
        },
    )

    def fail(*args, **kwargs):
        raise exception_type(message)

    def solved_builtin(*args, **kwargs):
        return torch.ones((2, 1), dtype=torch.float64), {
            "status": "solved",
            "status_val": 1,
            "status_compatible": True,
            "primal_infeasibility_certificate": None,
            "dual_infeasibility_certificate": None,
        }

    monkeypatch.setattr(adapter, "_solve_torch_osqp_path", fail)
    monkeypatch.setattr(adapter, "_solve_builtin_osqp_path", solved_builtin)
    with pytest.warns(RuntimeWarning, match=message):
        _, info = adapter.solve_osqp_torch_qp(
            *_problem(),
            torch.device("cpu"),
            True,
            {"algebra": "auto", "settings": {"return_info": True}},
            workspace,
        )

    failure = info["fallback"]
    assert info["status"] == "solved"
    assert info["status_val"] == 1
    assert failure["trigger"] == "exception"
    assert failure["exception_type"] == exception_type.__name__
    assert failure["message"] == message
    assert "torch_info" not in failure
    assert "status" not in failure
    assert info["primal_infeasibility_certificate"] is None
    assert info["dual_infeasibility_certificate"] is None


def test_auto_unsolved_status_records_status_and_falls_back(monkeypatch):
    problem = _problem()
    monkeypatch.setattr(
        adapter,
        "_select_backend",
        lambda *args, **kwargs: {
            "requested_backend": "auto",
            "backend": "torch",
            "selection_reason": "synthetic_validated_target",
            "estimated_kkt_dim": 4,
            "estimated_dense_working_memory_mb": 1.0,
            "fallback": {"occurred": False},
        },
    )

    def unsolved(*args, **kwargs):
        return torch.zeros((2, 1), dtype=torch.float64), {
            "status": "max_iter_reached",
            "status_compatible": False,
        }

    monkeypatch.setattr(adapter, "_solve_torch_osqp_path", unsolved)
    with pytest.warns(RuntimeWarning, match="solved-compatible"):
        _, info = adapter.solve_osqp_torch_qp(
            *problem,
            torch.device("cpu"),
            True,
            {"algebra": "auto", "settings": {"return_info": True}},
            TorchOSQPWorkspace(),
        )
    assert info["backend"] == "builtin"
    assert info["fallback"]["trigger"] == "unsolved_status"
    assert info["fallback"]["status"] == "max_iter_reached"
    assert info["fallback"]["torch_info"]["status_compatible"] is False


def test_unpromoted_cuda_auto_fallback_is_observable():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("An NVIDIA CUDA device is required.")
    H, f, A, b, lower, upper = _problem(device="cuda")
    with pytest.warns(RuntimeWarning, match="cuda_not_promoted"):
        _, info = adapter.solve_osqp_torch_qp(
            H,
            f,
            A,
            b,
            lower,
            upper,
            "cuda",
            True,
            {"algebra": "auto", "settings": {"return_info": True}},
            TorchOSQPWorkspace(),
        )
    assert info["backend"] == "builtin"
    assert info["selection_reason"] == "cuda_not_promoted"
    assert info["fallback"]["occurred"]
    assert info["fallback"]["trigger"] == "selection_policy"


def test_auto_memory_preflight_selects_observable_builtin_fallback(monkeypatch):
    monkeypatch.setattr(adapter, "_accelerator_capability", lambda *args: (True, "ok"))
    monkeypatch.setattr(adapter, "_memory_limit_mb", lambda *args: 0.0001)
    H, f, A, b, *_ = _problem(n=10)
    with pytest.warns(RuntimeWarning, match="memory"):
        selection = adapter._select_backend(
            "auto", torch.device("cuda"), torch.float64, H, f, A, b
        )
    assert selection["backend"] == "builtin"
    assert selection["selection_reason"] == "memory_preflight"
    assert selection["fallback"]["occurred"]


def test_mps_float64_auto_fallback_returns_cpu_solution():
    problem = _problem()
    with pytest.warns(RuntimeWarning, match="mps_float64_unsupported"):
        solution, info = adapter.solve_osqp_torch_qp(
            *problem,
            torch.device("mps"),
            True,
            {"algebra": "auto", "settings": {"return_info": True}},
            TorchOSQPWorkspace(),
        )
    assert solution.device.type == "cpu"
    assert info["requested_device"] == "mps"
    assert info["device"] == "cpu"
    assert info["selection_reason"] == "mps_float64_unsupported"
    assert info["fallback"]["occurred"]


def test_only_three_public_algebra_values_are_accepted():
    with pytest.raises(ValueError, match="auto.*builtin.*torch"):
        adapter._normalize_options({"algebra": "cuda"}, torch.float64)


def test_explicit_torch_unsolved_status_raises(monkeypatch):
    problem = _problem()

    def unsolved(*args, **kwargs):
        return torch.zeros((2, 1), dtype=torch.float64), {
            "status": "max_iter_reached",
            "status_compatible": False,
        }

    monkeypatch.setattr(adapter, "_solve_torch_osqp_path", unsolved)
    with pytest.raises(RuntimeError, match="max_iter_reached"):
        adapter.solve_osqp_torch_qp(
            *problem,
            torch.device("cpu"),
            True,
            {"algebra": "torch", "settings": {"return_info": True}},
            TorchOSQPWorkspace(),
        )


def test_explicit_torch_time_limit_preserves_previous_committed_warm_state(
    monkeypatch,
):
    workspace = TorchOSQPWorkspace()
    problem = _problem()
    adapter.solve_osqp_torch_qp(
        *problem,
        torch.device("cpu"),
        True,
        {
            "algebra": "torch",
            "settings": {
                "adaptive_rho": False,
                "polishing": False,
                "return_info": True,
            },
        },
        workspace,
    )
    state_before = {key: value.clone() for key, value in workspace.state.items()}
    clock_calls = 0

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls == 1 else 0.5

    monkeypatch.setattr(torch_osqp_module, "_clock", clock)
    changed = list(problem)
    changed[1] = -10.0 * torch.ones_like(changed[1])
    with pytest.raises(RuntimeError, match="time_limit_reached"):
        adapter.solve_osqp_torch_qp(
            *changed,
            torch.device("cpu"),
            True,
            {
                "algebra": "torch",
                "settings": {
                    "time_limit": 0.5,
                    "max_iter": 20,
                    "check_termination": 0,
                    "adaptive_rho": False,
                    "polishing": False,
                    "warm_start": False,
                    "eps_abs": 1e-12,
                    "eps_rel": 1e-12,
                    "return_info": True,
                },
            },
            workspace,
        )

    assert workspace.last_info["status"] == "time_limit_reached"
    assert workspace.last_info["status_val"] == 8
    assert workspace.last_info["status_compatible"] is False
    for key, value in state_before.items():
        assert torch.equal(workspace.state[key], value)


def test_backend_change_invalidates_complete_workspace():
    workspace = TorchOSQPWorkspace()
    workspace.state = {"x": torch.ones(1), "z": torch.ones(1), "y": torch.ones(1)}
    workspace.active_backend = "torch"
    assert workspace.ensure_backend("builtin")
    assert workspace.state is None
    assert workspace.builtin_cache is None
    assert workspace.active_backend == "builtin"


def test_unindexed_cuda_target_is_canonicalized_to_the_current_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")
    canonical = adapter._canonical_device(torch.device("cuda"))
    assert canonical.type == "cuda"
    assert canonical.index == torch.cuda.current_device()

def test_rho_update_uses_pinned_policy():
    assert adapter._normalize_options(None, torch.float64)["settings"]["rho_update_safeguard"] is False
    with pytest.raises(ValueError, match="rho_update_safeguard=False"):
        adapter._normalize_options({"settings": {"rho_update_safeguard": True}}, torch.float64)
