import importlib
import re
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

import bench_pygranso_osqp_workloads as benchmark_module
from bench_pygranso_osqp_workloads import (
    equivalent_metrics,
    make_workload,
    summarize_trace,
)
from pygranso.private.osqpTorchAdapter import _normalize_options
from pygranso.private.solveQP import beginOSQPTrace, endOSQPTrace
from pygranso.pygranso import pygranso
from pygranso.pygransoOptions import getDefaults
from pygranso.pygransoStruct import pygransoStruct


def test_default_torch_optimizer_reaches_known_constrained_solution(
    monkeypatch, floating_dtype, available_device
):
    adapter = importlib.import_module("pygranso.private.osqpTorchAdapter")
    original = adapter.solve_torch_osqp_direct
    calls = []

    def capture(P, q, A, lower, upper, settings, workspace=None):
        assert all(t.device == opts.x0.device for t in (P, q, A, lower, upper))
        assert all(t.dtype == floating_dtype for t in (P, q, A, lower, upper))
        assert not settings["use_sparse"]
        result = original(P, q, A, lower, upper, settings, workspace)
        calls.append(result[1]["status"])
        return result

    monkeypatch.setattr(adapter, "solve_torch_osqp_direct", capture)

    def objective(variables):
        x, y = variables.x[0], variables.x[1]
        inequalities, equalities = pygransoStruct(), pygransoStruct()
        inequalities.nonnegative_x = -x
        equalities.fixed_y = y - 2
        return (x + 1) ** 2 + (y - 3) ** 2, inequalities, equalities

    tolerance = 1e-4 if floating_dtype == torch.float32 else 1e-7
    opts = pygransoStruct()
    opts.torch_device = available_device
    opts.double_precision = floating_dtype == torch.float64
    opts.x0 = torch.tensor(
        [[-2.0], [-2.0]], device=available_device, dtype=floating_dtype
    )
    opts.QPsolver = "osqp"
    opts.print_level = 0
    opts.quadprog_info_msg = False
    opts.maxit = 100
    opts.opt_tol = tolerance
    opts.viol_eq_tol = tolerance
    opts.viol_ineq_tol = tolerance
    opts.osqp_settings = {
        "eps_abs": tolerance / 10,
        "eps_rel": tolerance / 10,
        "max_iter": 4000,
        "polishing": False,
    }
    solution = pygranso({"x": [2, 1]}, objective, opts)

    assert solution.termination_code == 0
    assert solution.quadprog_failure_rate == 0
    assert len(calls) > 2
    assert set(calls) <= {"solved", "solved_inaccurate"}
    expected = torch.tensor(
        [[0.0], [2.0]], device=available_device, dtype=floating_dtype
    )
    assert solution.final.x.device == opts.x0.device
    assert solution.final.x.dtype == floating_dtype
    assert torch.allclose(solution.final.x, expected, atol=tolerance, rtol=0)
    assert abs(float(solution.final.f) - 2.0) <= 4 * tolerance
    assert float(solution.final.tv) <= 2 * tolerance
    assert float(solution.stat_value) <= tolerance


def _run(name, algebra, *, capture_trace=False):
    args = Namespace(
        seed=0,
        maxit=1,
        eps_abs=1e-8,
        eps_rel=1e-8,
        qp_max_iter=4000,
    )
    method = "cpu_warm" if algebra == "builtin" else "torch_cuda"
    var_spec, combined_fn, opts = make_workload(
        name, "cpu", args.seed, args.maxit, method, args
    )
    opts.osqp_algebra = algebra
    if capture_trace:
        beginOSQPTrace(capture_data=True)
    try:
        solution = pygranso(
            var_spec=var_spec,
            combined_fn=combined_fn,
            user_opts=opts,
        )
    finally:
        trace = endOSQPTrace() if capture_trace else None
    metrics = {
        "termination_code": int(solution.termination_code),
        "objective": float(solution.final.f),
        "feasibility": float(solution.final.tv),
        "stationarity": float(solution.stat_value),
        "quadprog_failure_rate": float(solution.quadprog_failure_rate),
    }
    return metrics, trace


def test_explicit_torch_solve_qp_and_bfgssqp_never_import_python_osqp():
    repository = Path(__file__).resolve().parents[1]
    script = r"""
import importlib.abc
import sys
from argparse import Namespace


class BlockOSQP(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "osqp" or fullname.startswith("osqp."):
            raise AssertionError(f"unexpected Python OSQP import: {fullname}")
        return None


sys.meta_path.insert(0, BlockOSQP())

import torch
from bench_pygranso_osqp_workloads import make_workload
from pygranso.private.solveQP import beginOSQPTrace, endOSQPTrace, solveQP
from pygranso.pygranso import pygranso

for dtype, double_precision in ((torch.float32, False), (torch.float64, True)):
    H = torch.ones((1, 1), dtype=dtype)
    f = -torch.ones((1, 1), dtype=dtype)
    lower = torch.zeros((1, 1), dtype=dtype)
    upper = torch.full((1, 1), 2.0, dtype=dtype)
    qp_solution, qp_info = solveQP(
        H,
        f,
        None,
        None,
        lower,
        upper,
        "osqp",
        torch.device("cpu"),
        double_precision,
        {
            "algebra": "torch",
            "settings": {
                "adaptive_rho": False,
                "check_termination": 1,
                "polishing": False,
                "return_info": True,
                "scaling": 0,
            },
        },
    )
    assert qp_info["backend"] == "torch"
    assert qp_info["status"] == "solved"
    assert qp_solution.device.type == "cpu"
    assert qp_solution.dtype == dtype
    assert torch.allclose(qp_solution, torch.ones_like(qp_solution), atol=1e-4, rtol=1e-4)

args = Namespace(
    seed=0,
    maxit=1,
    eps_abs=1e-8,
    eps_rel=1e-8,
    qp_max_iter=4000,
)
for workload_name in ("B1", "B2", "B3"):
    var_spec, combined_fn, opts = make_workload(
        workload_name, "cpu", args.seed, args.maxit, "torch_cuda", args
    )
    opts.osqp_algebra = "torch"
    beginOSQPTrace(capture_data=False)
    try:
        solution = pygranso(
            var_spec=var_spec,
            combined_fn=combined_fn,
            user_opts=opts,
        )
    finally:
        trace = endOSQPTrace()

    assert len(trace) == 3
    assert solution.quadprog_failure_rate == 0.0
    assert torch.isfinite(torch.as_tensor(solution.final.f))
    assert torch.isfinite(torch.as_tensor(solution.final.tv))
    assert torch.isfinite(torch.as_tensor(solution.stat_value))
assert not any(name == "osqp" or name.startswith("osqp.") for name in sys.modules)
print("explicit-torch-solveqp-b1-b2-b3-no-osqp-ok")
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == "explicit-torch-solveqp-b1-b2-b3-no-osqp-ok"


def test_default_osqp_policy_uses_the_existing_torch_route():
    assert getDefaults(1)[0].osqp_algebra == "torch"
    assert _normalize_options(None, torch.float64)["algebra"] == "torch"


def test_python_osqp_is_an_optional_reference_dependency_not_a_runtime_dependency():
    repository = Path(__file__).resolve().parents[1]
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    with (repository / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]

    def package_name(requirement):
        return re.match(r"[A-Za-z0-9_.-]+", requirement).group(0).lower()

    runtime_names = {package_name(value) for value in project["dependencies"]}
    reference_names = {
        package_name(value)
        for value in project["optional-dependencies"]["reference"]
    }
    assert "osqp" not in runtime_names
    assert "osqp" in reference_names


def test_default_solve_qp_and_bfgssqp_never_import_python_osqp():
    repository = Path(__file__).resolve().parents[1]
    script = r"""
import importlib.abc
import sys
from argparse import Namespace


class BlockOSQP(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "osqp" or fullname.startswith("osqp."):
            raise AssertionError(f"unexpected Python OSQP import: {fullname}")
        return None


sys.meta_path.insert(0, BlockOSQP())

import torch
from bench_pygranso_osqp_workloads import make_workload
from pygranso.private.solveQP import beginOSQPTrace, endOSQPTrace, solveQP
from pygranso.pygranso import pygranso

for dtype, double_precision in ((torch.float32, False), (torch.float64, True)):
    H = torch.ones((1, 1), dtype=dtype)
    f = -torch.ones((1, 1), dtype=dtype)
    lower = torch.zeros((1, 1), dtype=dtype)
    upper = torch.full((1, 1), 2.0, dtype=dtype)
    qp_solution, qp_info = solveQP(
        H,
        f,
        None,
        None,
        lower,
        upper,
        "osqp",
        torch.device("cpu"),
        double_precision,
        {
            "settings": {
                "adaptive_rho": False,
                "check_termination": 1,
                "polishing": False,
                "return_info": True,
                "scaling": 0,
            },
        },
    )
    assert qp_info["backend"] == "torch"
    assert qp_info["requested_backend"] == "torch"
    assert qp_info["status"] == "solved"
    assert qp_solution.device.type == "cpu"
    assert qp_solution.dtype == dtype
    assert torch.allclose(qp_solution, torch.ones_like(qp_solution), atol=1e-4, rtol=1e-4)

args = Namespace(
    seed=0,
    maxit=1,
    eps_abs=1e-8,
    eps_rel=1e-8,
    qp_max_iter=4000,
)
for workload_name in ("B1", "B2", "B3"):
    var_spec, combined_fn, opts = make_workload(
        workload_name, "cpu", args.seed, args.maxit, "torch_cuda", args
    )
    delattr(opts, "osqp_algebra")
    beginOSQPTrace(capture_data=False)
    try:
        solution = pygranso(
            var_spec=var_spec,
            combined_fn=combined_fn,
            user_opts=opts,
        )
    finally:
        trace = endOSQPTrace()

    assert len(trace) == 3
    assert solution.quadprog_failure_rate == 0.0
    assert torch.isfinite(torch.as_tensor(solution.final.f))
    assert torch.isfinite(torch.as_tensor(solution.final.tv))
    assert torch.isfinite(torch.as_tensor(solution.stat_value))
assert not any(name == "osqp" or name.startswith("osqp.") for name in sys.modules)
print("default-torch-solveqp-b1-b2-b3-no-osqp-ok")
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == "default-torch-solveqp-b1-b2-b3-no-osqp-ok"


def test_pygranso_qp_failure_rate_is_isolated_to_the_current_run(monkeypatch):
    solve_qp_module = importlib.import_module("pygranso.private.solveQP")
    monkeypatch.setattr(solve_qp_module, "QP_REQUESTS", 10)
    monkeypatch.setattr(solve_qp_module, "QP_ERRORS", 7)

    metrics, _ = _run("B1", "torch")

    assert metrics["quadprog_failure_rate"] == 0.0
    requests, errors = solve_qp_module.getErr()
    assert requests > 0
    assert errors == 0


@pytest.mark.parametrize("workload", ["B1", "B2", "B3"])
def test_b1_b2_b3_complete_runs_agree_on_optimizer_observables(workload):
    builtin, trace = _run(workload, "builtin", capture_trace=True)
    torch_result, _ = _run(workload, "torch")
    assert torch_result["termination_code"] == builtin["termination_code"]
    for key in ("objective", "feasibility", "stationarity"):
        scale = max(1.0, abs(builtin[key]))
        assert abs(torch_result[key] - builtin[key]) / scale <= 1e-7

    summary = summarize_trace(trace)
    assert summary["qp_count"] >= 1
    assert summary["qp_matrix_value_changes"] >= 0
    assert summary["qp_shapes"]


def test_b2_benchmark_compares_identical_normalized_solver_settings():
    args = Namespace(
        seed=0,
        maxit=20,
        eps_abs=1e-5,
        eps_rel=1e-5,
        qp_max_iter=37,
    )
    _, _, builtin_opts = make_workload("B2", "cpu", 0, 20, "cpu_warm", args)
    _, _, torch_opts = make_workload("B2", "cpu", 0, 20, "torch_cuda", args)

    builtin_settings = _normalize_options(
        {"algebra": "builtin", "settings": builtin_opts.osqp_settings},
        torch.float64,
    )["settings"]
    torch_settings = _normalize_options(
        {"algebra": "torch", "settings": torch_opts.osqp_settings},
        torch.float64,
    )["settings"]

    assert builtin_settings == torch_settings
    assert builtin_settings["max_iter"] == 37


@pytest.mark.parametrize("key", ["objective", "feasibility", "stationarity"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_equivalent_metrics_rejects_nonfinite_values(key, bad_value):
    baseline = {
        "termination_code": 0,
        "objective": 1.0,
        "feasibility": 0.0,
        "stationarity": 0.0,
    }
    candidate = dict(baseline)
    candidate[key] = bad_value
    assert not equivalent_metrics(baseline, candidate)
    assert not equivalent_metrics(candidate, baseline)


def _benchmark_args(**updates):
    values = {
        "workloads": ["B1"],
        "repeats": 2,
        "warmups": 0,
        "maxit": 17,
        "seed": 5,
        "qp_max_iter": 37,
        "eps_abs": 1e-5,
        "eps_rel": 1e-5,
        "maximum_slowdown": 5.0,
    }
    values.update(updates)
    return Namespace(**values)


def _metrics(**updates):
    values = {
        "termination_code": 0,
        "objective": 1.0,
        "feasibility": 0.0,
        "stationarity": 0.0,
    }
    values.update(updates)
    return values


def test_benchmark_rejects_nonpositive_repeat_count_before_cuda_selection():
    with pytest.raises(ValueError, match="--repeats must be at least 1"):
        benchmark_module.benchmark(_benchmark_args(repeats=0))


def _stub_benchmark_runs(monkeypatch, method_metrics, events=None):
    positions = {method: 0 for method in method_metrics}

    def fake_run(_workload, method, _args):
        if events is not None:
            events.append(("run", method))
        position = positions[method]
        positions[method] += 1
        return (1.0 if method == "cpu_warm" else 2.0), method_metrics[method][
            position
        ]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark_module, "run_workload", fake_run)
    monkeypatch.setattr(benchmark_module, "solution_metrics", lambda result: result)
    monkeypatch.setattr(
        benchmark_module, "bootstrap_speedup_interval", lambda *_args: (0.5, 0.5)
    )


@pytest.mark.parametrize(
    "bad_metrics",
    [
        _metrics(termination_code=4),
        _metrics(stationarity=1e-3),
        _metrics(stationarity=float("nan")),
    ],
)
def test_benchmark_rejects_any_non_equivalent_measured_repeat(
    monkeypatch, bad_metrics
):
    baseline = _metrics()
    _stub_benchmark_runs(
        monkeypatch,
        {
            "cpu_warm": [dict(baseline), dict(baseline)],
            "torch_cuda": [bad_metrics, dict(baseline)],
        },
    )
    monkeypatch.setattr(
        benchmark_module,
        "trace_workload",
        lambda *_args: {
            "qp_count": 1,
            "qp_structure_changes": 0,
            "qp_matrix_value_changes": 0,
            "qp_shapes": "shape",
            "qp_trace_devices": "cpu",
        },
    )

    args = _benchmark_args()
    rows = benchmark_module.benchmark(args)
    torch_row = next(row for row in rows if row["device"] == "cuda")

    assert not torch_row["equivalent_to_cpu"]
    assert torch_row["performance_gate"] == "fail"
    assert args.gate_failed


def test_benchmark_attaches_each_backend_own_full_trace(monkeypatch):
    baseline = _metrics()
    events = []
    _stub_benchmark_runs(
        monkeypatch,
        {
            "cpu_warm": [dict(baseline), dict(baseline)],
            "torch_cuda": [dict(baseline), dict(baseline)],
        },
        events,
    )
    trace_calls = []

    def fake_trace(workload, method, args):
        events.append(("trace", method))
        trace_calls.append((workload, method, args.maxit))
        device = "cpu" if method == "cpu_warm" else "cuda:0"
        return {
            "qp_count": 11 if method == "cpu_warm" else 22,
            "qp_structure_changes": 1,
            "qp_matrix_value_changes": 2,
            "qp_shapes": "shape",
            "qp_trace_devices": device,
        }

    monkeypatch.setattr(benchmark_module, "trace_workload", fake_trace)

    rows = benchmark_module.benchmark(_benchmark_args())
    cpu_row = next(row for row in rows if row["device"] == "cpu")
    torch_row = next(row for row in rows if row["device"] == "cuda")

    assert trace_calls == [
        ("B1", "cpu_warm", 17),
        ("B1", "torch_cuda", 17),
    ]
    assert events == [
        ("run", "cpu_warm"),
        ("run", "cpu_warm"),
        ("run", "torch_cuda"),
        ("run", "torch_cuda"),
        ("trace", "cpu_warm"),
        ("trace", "torch_cuda"),
    ]
    assert cpu_row["qp_count"] == 11
    assert cpu_row["qp_trace_devices"] == "cpu"
    assert torch_row["qp_count"] == 22
    assert torch_row["qp_trace_devices"] == "cuda:0"
    assert torch_row["equivalent_to_cpu"]
    assert torch_row["performance_gate"] == "pass"


@pytest.mark.parametrize(
    ("method", "expected_device"),
    [("cpu_warm", "cpu"), ("torch_cuda", "cuda")],
)
def test_trace_workload_uses_requested_method_and_full_maxit(
    monkeypatch, method, expected_device
):
    calls = []
    trace = [
        {
            "H": {"shape": (2, 2), "device": expected_device},
            "A": None,
            "structure_changed": False,
            "matrix_values_changed": False,
        }
    ]
    monkeypatch.setattr(
        benchmark_module,
        "beginOSQPTrace",
        lambda capture_data: calls.append(("begin", capture_data)),
    )
    monkeypatch.setattr(benchmark_module, "endOSQPTrace", lambda: trace)
    monkeypatch.setattr(
        benchmark_module,
        "run_workload",
        lambda name, selected_method, args: calls.append(
            (name, selected_method, args.maxit)
        ),
    )

    summary = benchmark_module.trace_workload("B1", method, _benchmark_args())

    assert calls == [("begin", True), ("B1", method, 17)]
    assert summary["qp_trace_devices"] == expected_device


def test_run_workload_repeats_fallback_rng_and_restores_caller_state(monkeypatch):
    draws = []
    sentinel = object()
    monkeypatch.setattr(
        benchmark_module,
        "make_workload",
        lambda *_args: ({"x": [1, 1]}, lambda _variables: None, object()),
    )

    def fake_pygranso(**_kwargs):
        draws.append(torch.randn(4))
        return sentinel

    monkeypatch.setattr(benchmark_module, "pygranso", fake_pygranso)
    torch.manual_seed(91)
    state_before_first = torch.random.get_rng_state()
    _, first = benchmark_module.run_workload(
        "B1", "cpu_warm", _benchmark_args()
    )
    state_after_first = torch.random.get_rng_state()
    torch.randn(7)
    state_before_second = torch.random.get_rng_state()
    _, second = benchmark_module.run_workload(
        "B1", "cpu_warm", _benchmark_args()
    )

    assert first is sentinel
    assert second is sentinel
    assert torch.equal(draws[0], draws[1])
    assert torch.equal(state_before_first, state_after_first)
    assert torch.equal(state_before_second, torch.random.get_rng_state())


def test_run_workload_restores_caller_rng_when_pygranso_raises(monkeypatch):
    monkeypatch.setattr(
        benchmark_module,
        "make_workload",
        lambda *_args: ({"x": [1, 1]}, lambda _variables: None, object()),
    )

    def raise_after_draw(**_kwargs):
        torch.randn(4)
        raise RuntimeError("expected benchmark failure")

    monkeypatch.setattr(benchmark_module, "pygranso", raise_after_draw)
    torch.manual_seed(92)
    state_before = torch.random.get_rng_state()

    with pytest.raises(RuntimeError, match="expected benchmark failure"):
        benchmark_module.run_workload("B1", "cpu_warm", _benchmark_args())

    assert torch.equal(state_before, torch.random.get_rng_state())


def test_trace_workload_ends_trace_when_run_raises(monkeypatch):
    events = []
    monkeypatch.setattr(
        benchmark_module,
        "beginOSQPTrace",
        lambda capture_data: events.append(("begin", capture_data)),
    )
    monkeypatch.setattr(
        benchmark_module,
        "endOSQPTrace",
        lambda: events.append(("end", None)) or [],
    )

    def fail_run(*_args):
        events.append(("run", None))
        raise RuntimeError("expected trace failure")

    monkeypatch.setattr(benchmark_module, "run_workload", fail_run)

    with pytest.raises(RuntimeError, match="expected trace failure"):
        benchmark_module.trace_workload("B1", "cpu_warm", _benchmark_args())

    assert events == [("begin", True), ("run", None), ("end", None)]
