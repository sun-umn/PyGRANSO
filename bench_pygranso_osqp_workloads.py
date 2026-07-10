import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from bench_osqp_dense_reference import bootstrap_speedup_interval, markdown_table
from pygranso.private.solveQP import beginOSQPTrace, endOSQPTrace
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

WORKLOADS = ("B1", "B2", "B3")
RESULT_COLUMNS = (
    "method",
    "workload",
    "device",
    "median_ms",
    "iqr_ms",
    "speedup_vs_cpu_warm",
    "speedup_ci_low",
    "speedup_ci_high",
    "torch_slowdown",
    "performance_gate",
    "termination_code",
    "objective",
    "feasibility",
    "stationarity",
    "equivalent_to_cpu",
    "qp_count",
    "qp_structure_changes",
    "qp_matrix_value_changes",
    "qp_shapes",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="End-to-end CPU OSQP vs Torch CUDA PyGRANSO workloads."
    )
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=WORKLOADS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--maxit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qp-max-iter", type=int, default=4000)
    parser.add_argument("--eps-abs", type=float, default=1e-5)
    parser.add_argument("--eps-rel", type=float, default=1e-5)
    parser.add_argument("--export-md", default=None)
    parser.add_argument("--export-csv", default=None)
    parser.add_argument("--maximum-slowdown", type=float, default=5.0)
    parser.add_argument("--enforce-gate", action="store_true")
    return parser.parse_args(argv)


def make_workload(name, device, seed, maxit, method, args):
    device = torch.device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    opts = pygransoStruct()
    opts.torch_device = device
    opts.print_level = 0
    opts.quadprog_info_msg = False
    opts.maxit = int(maxit)
    opts.QPsolver = "osqp"

    if name == "B1":
        var_spec = {"x": [1, 1], "y": [1, 1]}

        def combined_fn(variables):
            x = variables.x
            y = variables.y
            inequalities = pygransoStruct()
            inequalities.c1 = (y + x**2) ** 2 + 0.1 * y**2 - 1
            inequalities.c2 = y - torch.exp(-x) - 3
            inequalities.c3 = y - x + 4
            return 0 * x + 0 * y, inequalities, None

        opts.x0 = torch.zeros((2, 1), device=device, dtype=torch.double)
    elif name == "B2":
        n = 300
        A = torch.randn((n, n), generator=generator, dtype=torch.double)
        A = (0.5 * (A + A.T)).to(device=device)
        x0 = torch.randn((n, 1), generator=generator, dtype=torch.double).to(
            device=device
        )
        var_spec = {"x": [n, 1]}

        def combined_fn(variables):
            x = variables.x
            equalities = pygransoStruct()
            equalities.c1 = x.T @ x - 1
            return -x.T @ A @ x, None, equalities

        opts.x0 = x0
        opts.mu0 = 0.1
        opts.opt_tol = 1e-6
    else:
        n = 5
        d = 1
        A = torch.randn((n, n), generator=generator, dtype=torch.double)
        A = (0.5 * (A + A.T)).to(device=device)
        x0 = torch.randn((n * d, 1), generator=generator, dtype=torch.double).to(
            device=device
        )
        var_spec = {"V": [n, d]}

        def combined_fn(variables):
            V = variables.V
            equalities = pygransoStruct()
            equalities.c1 = V.T @ V - torch.eye(
                d, device=device, dtype=torch.double
            )
            return -torch.trace(V.T @ A @ V), None, equalities

        opts.x0 = x0

    if method == "cpu_warm":
        opts.osqp_algebra = "builtin"
        opts.osqp_settings = {
            "eps_abs": args.eps_abs,
            "eps_rel": args.eps_rel,
            "polishing": False,
            "verbose": False,
        }
    else:
        opts.osqp_algebra = "torch"
        opts.osqp_settings = {
            "max_iter": args.qp_max_iter,
            "check_termination": min(25, args.qp_max_iter),
            "eps_abs": args.eps_abs,
            "eps_rel": args.eps_rel,
            "warm_start": True,
            "adaptive_rho": True,
            "rho_update_interval": 50,
            "scaling": 10,
            "polishing": True,
            "verbose": False,
        }
    return var_spec, combined_fn, opts


def run_workload(name, method, args):
    device = "cpu" if method == "cpu_warm" else "cuda"
    var_spec, combined_fn, opts = make_workload(
        name, device, args.seed, args.maxit, method, args
    )
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    solution = pygranso(var_spec=var_spec, combined_fn=combined_fn, user_opts=opts)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, solution


def solution_metrics(solution):
    final = solution.final
    return {
        "termination_code": int(solution.termination_code),
        "objective": float(torch.as_tensor(final.f).item()),
        "feasibility": float(torch.as_tensor(final.tv).item()),
        "stationarity": float(solution.stat_value),
    }


def trace_workload(name, args):
    var_spec, combined_fn, opts = make_workload(
        name, "cpu", args.seed, min(args.maxit, 5), "cpu_warm", args
    )
    beginOSQPTrace(capture_data=True)
    try:
        pygranso(var_spec=var_spec, combined_fn=combined_fn, user_opts=opts)
    finally:
        trace = endOSQPTrace()
    return summarize_trace(trace)


def summarize_trace(trace):
    shapes = sorted(
        {
            str((record["H"]["shape"], None if record["A"] is None else record["A"]["shape"]))
            for record in trace
        }
    )
    return {
        "qp_count": len(trace),
        "qp_structure_changes": sum(
            bool(record["structure_changed"]) for record in trace
        ),
        "qp_matrix_value_changes": sum(
            bool(record["matrix_values_changed"]) for record in trace
        ),
        "qp_shapes": "; ".join(shapes),
    }


def equivalent_metrics(cpu, cuda, tolerance=1e-4):
    if cpu["termination_code"] != cuda["termination_code"]:
        return False
    for key in ("objective", "feasibility", "stationarity"):
        scale = max(1.0, abs(cpu[key]))
        if abs(cpu[key] - cuda[key]) / scale > tolerance:
            return False
    return True


def benchmark(args):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the real PyGRANSO comparison.")
    rows = []
    gate_failed = False
    for workload in args.workloads:
        trace = trace_workload(workload, args)
        method_data = {}
        for method in ("cpu_warm", "torch_cuda"):
            for _ in range(args.warmups):
                run_workload(workload, method, args)
            timings = []
            last_solution = None
            for _ in range(args.repeats):
                elapsed_ms, last_solution = run_workload(workload, method, args)
                timings.append(elapsed_ms)
            method_data[method] = {
                "timings": timings,
                "metrics": solution_metrics(last_solution),
            }

        cpu = method_data["cpu_warm"]
        cuda = method_data["torch_cuda"]
        ci = bootstrap_speedup_interval(cpu["timings"], cuda["timings"])
        cpu_median = statistics.median(cpu["timings"])
        cuda_median = statistics.median(cuda["timings"])
        equivalent = equivalent_metrics(cpu["metrics"], cuda["metrics"])
        slowdown = cuda_median / cpu_median
        passed = equivalent and slowdown <= args.maximum_slowdown
        gate_failed |= not passed
        for method, data in method_data.items():
            timings = np.asarray(data["timings"], dtype=float)
            row = {
                "method": "CPU OSQP update/warm"
                if method == "cpu_warm"
                else "Torch CUDA dense LU",
                "workload": workload,
                "device": "cpu" if method == "cpu_warm" else "cuda",
                "median_ms": float(np.median(timings)),
                "iqr_ms": float(np.percentile(timings, 75) - np.percentile(timings, 25)),
                "speedup_vs_cpu_warm": 1.0 if method == "cpu_warm" else cpu_median / cuda_median,
                "speedup_ci_low": 1.0 if method == "cpu_warm" or ci is None else ci[0],
                "speedup_ci_high": 1.0 if method == "cpu_warm" or ci is None else ci[1],
                "torch_slowdown": 1.0 if method == "cpu_warm" else slowdown,
                "performance_gate": (
                    "baseline" if method == "cpu_warm" else "pass" if passed else "fail"
                ),
                "equivalent_to_cpu": True if method == "cpu_warm" else equivalent,
                **data["metrics"],
                **trace,
            }
            rows.append(row)
    args.gate_failed = gate_failed
    return rows


def export_rows(rows, args):
    if args.export_md:
        path = Path(args.export_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown_table(rows, RESULT_COLUMNS) + "\n", encoding="utf-8")
    if args.export_csv:
        path = Path(args.export_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerows({key: row[key] for key in RESULT_COLUMNS} for row in rows)


def main(argv=None):
    args = parse_args(argv)
    rows = benchmark(args)
    print(markdown_table(rows, RESULT_COLUMNS))
    export_rows(rows, args)
    if args.enforce_gate and args.gate_failed:
        raise SystemExit(
            "Torch CUDA exceeded the end-to-end correctness/performance gate."
        )


if __name__ == "__main__":
    main()
