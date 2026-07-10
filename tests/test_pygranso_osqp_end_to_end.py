from argparse import Namespace

import pytest

from bench_pygranso_osqp_workloads import make_workload, summarize_trace
from pygranso.private.solveQP import beginOSQPTrace, endOSQPTrace
from pygranso.pygranso import pygranso


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
    }
    return metrics, trace


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
