import importlib
import warnings

import pytest
import torch

from pygranso.private import osqpTorchAdapter as adapter
from pygranso.private.bfgsHessianInverse import bfgsHessianInverse
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.qpSteeringStrategy import qpSS
from pygranso.private.qpTerminationCondition import qpTC
from pygranso.private.solveQP import (
    beginOSQPTrace,
    endOSQPTrace,
    getLastOSQPInfo,
    resetOSQPWarmState,
    solveQP,
)
from pygranso.private.torchLinearSolve import TorchLinearSolveError


def _problem(dtype=torch.float64):
    return (
        torch.eye(2, dtype=dtype),
        -torch.ones((2, 1), dtype=dtype),
        None,
        None,
        torch.zeros((2, 1), dtype=dtype),
        torch.ones((2, 1), dtype=dtype),
    )


def _solve_osqp(
    problem=None,
    *,
    dtype=torch.float64,
    double_precision=True,
    algebra="torch",
    settings=None,
    workspace=None,
):
    options = {"algebra": algebra}
    if settings is not None:
        options["settings"] = settings
    if workspace is not None:
        options["workspace"] = workspace
    return solveQP(
        *(problem if problem is not None else _problem(dtype)),
        "osqp",
        torch.device("cpu"),
        double_precision,
        options,
    )


def test_solve_qp_counts_an_osqp_exception_as_a_failed_request(monkeypatch):
    solve_qp_module = importlib.import_module("pygranso.private.solveQP")
    monkeypatch.setattr(solve_qp_module, "QP_REQUESTS", 0)
    monkeypatch.setattr(solve_qp_module, "QP_ERRORS", 0)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic counted QP failure")

    monkeypatch.setattr(solve_qp_module, "solve_osqp_torch_qp", fail)
    with pytest.raises(RuntimeError, match="synthetic counted QP failure"):
        solve_qp_module.solveQP(
            *_problem(),
            "osqp",
            torch.device("cpu"),
            True,
            {"algebra": "torch"},
        )

    assert solve_qp_module.getErr() == [1, 1]


def test_qp_counter_reset_and_keyboard_interrupt_separation(monkeypatch):
    solve_qp_module = importlib.import_module("pygranso.private.solveQP")
    monkeypatch.setattr(solve_qp_module, "QP_REQUESTS", 7)
    monkeypatch.setattr(solve_qp_module, "QP_ERRORS", 3)
    solve_qp_module.resetQPCounters()
    assert solve_qp_module.getErr() == [0, 0]

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("synthetic uncounted interrupt")

    monkeypatch.setattr(solve_qp_module, "solve_osqp_torch_qp", interrupt)
    with pytest.raises(KeyboardInterrupt, match="synthetic uncounted interrupt"):
        solve_qp_module.solveQP(
            *_problem(),
            "osqp",
            torch.device("cpu"),
            True,
            {"algebra": "torch"},
        )

    assert solve_qp_module.getErr() == [1, 0]


def test_solve_qp_rejects_nonboolean_return_info_before_workspace_mutation():
    workspace = TorchOSQPWorkspace()
    _solve_osqp(workspace=workspace)
    original_info = workspace.last_info
    original_warm_state = workspace.state

    with pytest.raises(ValueError, match="return_info must be a bool"):
        _solve_osqp(
            settings={"return_info": "false"},
            workspace=workspace,
        )

    assert workspace.last_info is original_info
    assert workspace.state is original_warm_state


def test_solve_qp_trace_can_omit_payload_and_preserves_caller_options():
    problem = _problem()
    options = {
        "algebra": "torch",
        "settings": {"return_info": False, "polishing": False},
    }
    expected_options = {
        "algebra": "torch",
        "settings": {"return_info": False, "polishing": False},
    }

    beginOSQPTrace(capture_data=False)
    try:
        solution = solveQP(
            *problem,
            "osqp",
            torch.device("cpu"),
            True,
            options,
        )
    finally:
        trace = endOSQPTrace()

    assert torch.is_tensor(solution)
    assert options == expected_options
    assert len(trace) == 1
    assert "qp" not in trace[0]
    assert trace[0]["H"] == {
        "shape": (2, 2),
        "layout": "torch.strided",
        "dtype": "torch.float64",
        "device": "cpu",
        "nnz": 2,
    }


def test_stationarity_qp_exhaustion_returns_outer_fallback_sentinel():
    termination = qpTC()
    termination.H = torch.eye(2, dtype=torch.float64)
    termination.all_grads = torch.eye(2, dtype=torch.float64)

    def fail(_matrix):
        raise RuntimeError("synthetic stationarity QP failure")

    termination.solveQP_fn = fail
    x, lambdas, stat_type, errors = termination.solveQPRobust(torch.float64)
    assert x is None
    assert lambdas is None
    assert stat_type == 0
    assert len(errors) == 3
    assert all("synthetic" in str(error) for error in errors)


def test_steering_qp_preserves_original_failure(monkeypatch):
    steering_module = importlib.import_module(
        "pygranso.private.qpSteeringStrategy"
    )
    steering = qpSS()
    steering.QPsolver = "osqp"
    steering.H = torch.eye(1, dtype=torch.float64)
    steering.f = torch.zeros((1, 1), dtype=torch.float64)
    steering.LB = -torch.ones((1, 1), dtype=torch.float64)
    steering.UB = torch.ones((1, 1), dtype=torch.float64)
    steering.device = torch.device("cpu")
    steering.double_precision = True
    steering.osqp_options = {}

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic steering QP failure")

    monkeypatch.setattr(steering_module, "solveQP", fail)
    with pytest.raises(RuntimeError, match="synthetic steering QP failure"):
        steering.solveSteeringDualQP()


@pytest.mark.parametrize(
    "double_precision,dtype",
    [(False, torch.float32), (True, torch.float64)],
)
def test_solved_inaccurate_metadata_trace_device_dtype_and_reset(
    monkeypatch, double_precision, dtype
):
    workspace = TorchOSQPWorkspace()
    problem = _problem(dtype)

    def solved_inaccurate(
        H,
        f,
        A,
        b,
        lower,
        upper,
        target_device,
        solve_dtype,
        settings,
        active_workspace,
        allow_device_move,
    ):
        assert settings["return_info"] is True
        assert not allow_device_move
        active_workspace.state = {
            "x": torch.full((f.numel(),), 0.5, device=target_device, dtype=solve_dtype)
        }
        return torch.full(
            (f.numel(), 1), 0.5, device=target_device, dtype=solve_dtype
        ), {
            "status": "solved_inaccurate",
            "status_val": 2,
            "status_compatible": True,
            "primal_infeasibility_certificate": None,
            "dual_infeasibility_certificate": None,
            "device": str(target_device),
            "dtype": str(solve_dtype),
        }

    monkeypatch.setattr(adapter, "_solve_torch_osqp_path", solved_inaccurate)
    beginOSQPTrace(capture_data=True)
    try:
        solution, info = _solve_osqp(
            problem,
            dtype=dtype,
            double_precision=double_precision,
            settings={"return_info": True},
            workspace=workspace,
        )
    finally:
        trace = endOSQPTrace()

    assert info["status"] == "solved_inaccurate"
    assert info["status_val"] == 2
    assert info["status_compatible"]
    assert info["primal_infeasibility_certificate"] is None
    assert info["dual_infeasibility_certificate"] is None
    assert info["device"] == "cpu"
    assert info["dtype"] == str(dtype)
    assert solution.device.type == "cpu"
    assert solution.dtype == dtype
    assert getLastOSQPInfo(workspace) is info
    assert len(trace) == 1
    assert trace[0]["H"]["device"] == "cpu"
    assert trace[0]["H"]["dtype"] == str(dtype)
    assert trace[0]["qp"][0].device.type == "cpu"
    assert trace[0]["qp"][0].dtype == dtype
    problem[0].zero_()
    assert torch.equal(trace[0]["qp"][0], torch.eye(2, dtype=dtype))

    resetOSQPWarmState(workspace)
    assert workspace.state is None
    assert workspace.last_info is None
    assert workspace.active_backend is None


@pytest.mark.parametrize("family", ["primal", "dual"])
def test_explicit_torch_certificate_preserves_info_and_warm_state(family):
    dtype = torch.float64
    workspace = TorchOSQPWorkspace()
    settings = {
        "return_info": True,
        "polishing": False,
        "check_termination": 1,
        "scaling": 0,
        "adaptive_rho": False,
    }
    if family == "primal":
        H = torch.ones((1, 1), dtype=dtype)
        f = torch.zeros((1, 1), dtype=dtype)
        A = torch.ones((1, 1), dtype=dtype)
        lower = torch.zeros((1, 1), dtype=dtype)
        upper = torch.ones((1, 1), dtype=dtype)
        feasible = (H, f, A, torch.tensor(0.5, dtype=dtype), lower, upper)
        infeasible = (H, f, A, torch.tensor(2.0, dtype=dtype), lower, upper)
        certificate_key, alias_key, status_values = (
            "primal_infeasibility_certificate",
            "prim_inf_cert",
            {3, 4},
        )
    else:
        lower = torch.full((1, 1), -torch.inf, dtype=dtype)
        upper = torch.full((1, 1), torch.inf, dtype=dtype)
        feasible = (
            torch.ones((1, 1), dtype=dtype),
            torch.zeros((1, 1), dtype=dtype),
            None,
            None,
            lower,
            upper,
        )
        infeasible = (
            torch.zeros((1, 1), dtype=dtype),
            -torch.ones((1, 1), dtype=dtype),
            None,
            None,
            lower,
            upper,
        )
        certificate_key, alias_key, status_values = (
            "dual_infeasibility_certificate",
            "dual_inf_cert",
            {5, 6},
        )

    _solve_osqp(feasible, settings=settings, workspace=workspace)
    prior_state = {key: value.clone() for key, value in workspace.state.items()}
    with pytest.raises(RuntimeError, match=rf"{family}_infeasible"):
        _solve_osqp(infeasible, settings=settings, workspace=workspace)

    info = workspace.last_info
    certificate = info[certificate_key]
    assert info["status"].startswith(f"{family}_infeasible")
    assert info["status_val"] in status_values
    assert not info["status_compatible"]
    assert certificate is not None
    assert torch.equal(info[alias_key], certificate)
    assert workspace.active_backend == "torch"
    assert set(workspace.state) == set(prior_state)
    for key, value in prior_state.items():
        assert torch.equal(workspace.state[key], value)


def test_auto_fallback_preserves_first_torch_status_and_certificate(monkeypatch):
    workspace = TorchOSQPWorkspace()
    certificate = torch.tensor([1.0, -1.0], dtype=torch.float64)

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

    def infeasible(*args, **kwargs):
        return torch.full((2, 1), torch.nan, dtype=torch.float64), {
            "status": "dual_infeasible",
            "status_val": 5,
            "status_compatible": False,
            "primal_infeasibility_certificate": None,
            "dual_infeasibility_certificate": certificate,
            "prim_inf_cert": None,
            "dual_inf_cert": certificate,
        }

    def solved_builtin(*args, **kwargs):
        return torch.ones((2, 1), dtype=torch.float64), {
            "status": "solved",
            "status_val": 1,
            "status_compatible": True,
            "backend": "builtin",
            "fallback": {"occurred": False},
        }

    monkeypatch.setattr(adapter, "_solve_torch_osqp_path", infeasible)
    monkeypatch.setattr(adapter, "_solve_builtin_osqp_path", solved_builtin)
    with pytest.warns(RuntimeWarning, match="solved-compatible"):
        solution, info = _solve_osqp(
            algebra="auto",
            settings={"return_info": True},
            workspace=workspace,
        )

    first_attempt = info["fallback"]["torch_info"]
    assert torch.equal(solution, torch.ones_like(solution))
    assert info["status"] == "solved"
    assert info["fallback"]["status"] == "dual_infeasible"
    assert first_attempt["status"] == "dual_infeasible"
    assert first_attempt["status_val"] == 5
    assert torch.equal(first_attempt["dual_infeasibility_certificate"], certificate)
    assert torch.equal(first_attempt["dual_inf_cert"], certificate)
    assert getLastOSQPInfo(workspace) is info


def test_keyboard_interrupt_propagates_without_replacing_workspace(monkeypatch):
    workspace = TorchOSQPWorkspace()
    workspace.active_backend = "torch"
    workspace.state = {"x": torch.tensor([0.25, 0.75], dtype=torch.float64)}
    workspace.last_info = {"status": "solved"}
    prior_state = {key: value.clone() for key, value in workspace.state.items()}
    prior_info = workspace.last_info

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("synthetic interrupt")

    monkeypatch.setattr(adapter, "_solve_torch_osqp_path", interrupt)
    with pytest.raises(KeyboardInterrupt, match="synthetic interrupt"):
        _solve_osqp(workspace=workspace)

    assert workspace.last_info is prior_info
    assert workspace.active_backend == "torch"
    assert set(workspace.state) == set(prior_state)
    for key, value in prior_state.items():
        assert torch.equal(workspace.state[key], value)


@pytest.mark.parametrize("layout", ["dense", "coo", "csr", "csc"])
def test_nonconvex_kkt_failure_preserves_valid_workspace(
    layout, floating_dtype, available_device, monkeypatch,
):
    H = torch.eye(2, dtype=floating_dtype, device=available_device)
    q = H.new_zeros((2, 1))
    lower, upper = -H.new_ones((2, 1)), H.new_ones((2, 1))
    workspace = TorchOSQPWorkspace()
    options = {
        "algebra": "torch",
        "workspace": workspace,
        "settings": {
            "return_info": True,
            "scaling": 10,
            "check_convexity": False,
            "polishing": False,
            "adaptive_rho": False,
        },
    }

    def solve(matrix):
        return solveQP(
            matrix, q, None, None, lower, upper, "osqp", available_device,
            floating_dtype == torch.float64, options,
        )

    _, initial_info = solve(H)
    assert initial_info["status"] == "solved"
    fields_before = workspace.__dict__.copy()
    factors_before = workspace.linear_solver.__dict__.copy()
    invalid = H.clone()
    invalid[1, 1] = -1.0
    if layout == "coo":
        invalid = invalid.to_sparse()
    elif layout in {"csr", "csc"}:
        # PyTorch emits its existing beta notice on first compressed conversion.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Sparse CSR tensor support is in beta state",
                category=UserWarning,
            )
            invalid = getattr(invalid, f"to_sparse_{layout}")()

    def forbid_builtin_or_host(*args, **kwargs):
        pytest.fail("Explicit Torch validation must not fall back or move to CPU")

    monkeypatch.setattr(adapter, "_solve_builtin_osqp_path", forbid_builtin_or_host)
    monkeypatch.setattr(torch.Tensor, "cpu", forbid_builtin_or_host)
    monkeypatch.setattr(torch.Tensor, "numpy", forbid_builtin_or_host)
    with pytest.raises(TorchLinearSolveError, match="KKT inertia"):
        if layout == "dense":
            solve(invalid)
        else:
            with pytest.warns(RuntimeWarning, match="dense compatibility"):
                solve(invalid)
    for name, value in fields_before.items():
        if name == "builtin_stats":
            assert workspace.builtin_stats == value
        else:
            assert workspace.__dict__[name] is value
    for name, value in factors_before.items():
        assert workspace.linear_solver.__dict__[name] is value
    assert getLastOSQPInfo(workspace) is initial_info

    solution, reused = solve(H)
    assert reused["status"] == "solved"
    assert reused["factorization_reused"]
    assert reused["factorizations_this_solve"] == 0
    assert solution.device == H.device and solution.dtype == H.dtype
    assert torch.equal(solution, q)


def test_nonfinite_bfgs_update_does_not_poison_next_torch_qp():
    initial = torch.eye(2, dtype=torch.float64)
    bfgs = bfgsHessianInverse(initial.clone(), False)
    step = torch.tensor([[1e-200], [0.0]], dtype=torch.float64)
    gradient_change = torch.tensor([[1e200], [0.0]], dtype=torch.float64)
    sty = step.T @ gradient_change

    assert bfgs.update(step, gradient_change, sty) == 3
    assert torch.equal(bfgs.getState(), initial)
    counts = bfgs.getCounts()
    assert counts.updates == 0
    assert counts.infnan_fails == 1

    solution, info = _solve_osqp(
        (bfgs.getState(), *_problem()[1:]),
        settings={
            "check_convexity": True,
            "polishing": False,
            "return_info": True,
        },
    )
    assert info["status_compatible"]
    assert torch.all(torch.isfinite(solution))


def test_indefinite_stationarity_qp_uses_existing_identity_fallback():
    termination = qpTC()
    termination.H = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
    termination.all_grads = torch.eye(2, dtype=torch.float64)
    attempts = []

    def solve(matrix):
        attempts.append(matrix.clone())
        return _solve_osqp(
            (matrix, *_problem()[1:]),
            settings={"check_convexity": True, "polishing": False},
        )

    termination.solveQP_fn = solve
    solution, _lambdas, stat_type, errors = termination.solveQPRobust(torch.float64)
    assert stat_type == 2
    assert len(errors) == 1
    assert "positive semidefinite" in str(errors[0])
    assert len(attempts) == 2
    assert torch.equal(attempts[0], termination.H)
    assert torch.equal(attempts[1], torch.eye(2, dtype=torch.float64))
    assert torch.all(torch.isfinite(solution))
