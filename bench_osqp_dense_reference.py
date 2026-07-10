"""Correctness and performance benchmark for the dense Torch OSQP reference."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from pygranso.private.osqpTorchAdapter import (
    MAX_SUPPORTED_KKT_DIM,
    _memory_limit_mb,
    estimate_dense_kkt,
    solve_osqp_torch_qp,
)
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace


def markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join((header, divider, *body))


def bootstrap_speedup_interval(baseline, candidate, samples=2000, seed=0):
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if baseline.size == 0 or candidate.size == 0:
        return None, None
    generator = np.random.default_rng(seed)
    ratios = []
    for _ in range(samples):
        b = generator.choice(baseline, baseline.size, replace=True)
        c = generator.choice(candidate, candidate.size, replace=True)
        ratios.append(np.median(b) / np.median(c))
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def make_case(name, n, device, dtype, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if name == "bound":
        H = torch.eye(n, dtype=dtype)
        f = -torch.ones((n, 1), dtype=dtype)
        A = b = None
    elif name == "equality":
        H = torch.eye(n, dtype=dtype)
        f = torch.zeros((n, 1), dtype=dtype)
        A = torch.ones((1, n), dtype=dtype)
        b = torch.tensor(1.0, dtype=dtype)
    elif name == "random_spd":
        rank = min(32, n)
        R = torch.randn((rank, n), generator=generator, dtype=dtype)
        H = R.T @ R + 1e-3 * torch.eye(n, dtype=dtype)
        f = torch.randn((n, 1), generator=generator, dtype=dtype)
        A = b = None
    else:
        raise ValueError(f"Unknown benchmark case {name!r}.")
    lower = -torch.ones((n, 1), dtype=dtype)
    upper = torch.ones((n, 1), dtype=dtype)
    values = (H, f, A, b, lower, upper)
    return tuple(
        value.to(device=device) if torch.is_tensor(value) else value for value in values
    )


def time_backend(problem, algebra, repeats, warmups):
    H, f, A, b, lower, upper = problem
    workspace = TorchOSQPWorkspace()
    settings = {
        "return_info": True,
        "eps_abs": 1e-8 if f.dtype == torch.float64 else 1e-5,
        "eps_rel": 1e-8 if f.dtype == torch.float64 else 1e-5,
    }
    timings = []
    last_info = None
    for index in range(warmups + repeats):
        _synchronize(f.device)
        started = time.perf_counter()
        _solution, info = solve_osqp_torch_qp(
            H,
            f,
            A,
            b,
            lower,
            upper,
            f.device,
            f.dtype == torch.float64,
            {"algebra": algebra, "settings": settings},
            workspace,
        )
        _synchronize(f.device)
        elapsed = (time.perf_counter() - started) * 1000
        if index >= warmups:
            timings.append(elapsed)
        last_info = info
    return timings, last_info


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def run(args):
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    rows = []
    gate_failed = False
    for case in args.cases:
        for n in args.sizes:
            problem = make_case(case, n, args.device, dtype, args.seed)
            H, f, A, b, _lower, _upper = problem
            kkt_dim, memory_mb = estimate_dense_kkt(H, f, A, b, dtype)
            if kkt_dim > MAX_SUPPORTED_KKT_DIM:
                raise SystemExit(
                    f"Case {case}/n={n} has KKT dimension {kkt_dim}, above "
                    f"the validated limit {MAX_SUPPORTED_KKT_DIM}."
                )
            if memory_mb > _memory_limit_mb(torch.device(args.device)):
                raise SystemExit(
                    f"Case {case}/n={n} needs an estimated {memory_mb:.1f} MiB, "
                    "above the conservative memory preflight."
                )
            builtin_times, builtin_info = time_backend(
                problem, "builtin", args.repeats, args.warmups
            )
            torch_times, torch_info = time_backend(
                problem, "torch", args.repeats, args.warmups
            )
            builtin_median = statistics.median(builtin_times)
            torch_median = statistics.median(torch_times)
            slowdown = torch_median / builtin_median
            ci_low, ci_high = bootstrap_speedup_interval(builtin_times, torch_times)
            passed = slowdown <= args.maximum_slowdown
            gate_failed |= not passed
            rows.append(
                {
                    "case": case,
                    "n": n,
                    "device": args.device,
                    "dtype": args.dtype,
                    "builtin_ms": f"{builtin_median:.3f}",
                    "torch_ms": f"{torch_median:.3f}",
                    "torch_slowdown": f"{slowdown:.3f}",
                    "speedup_ci_low": "" if ci_low is None else f"{ci_low:.3f}",
                    "speedup_ci_high": "" if ci_high is None else f"{ci_high:.3f}",
                    "performance_gate": "pass" if passed else "fail",
                    "builtin_status": builtin_info["status"],
                    "torch_status": torch_info["status"],
                    "primal_residual": f"{torch_info['primal_residual']:.3e}",
                    "dual_residual": f"{torch_info['dual_residual']:.3e}",
                }
            )
    columns = list(rows[0]) if rows else []
    print(markdown_table(rows, columns))
    if args.export_csv:
        path = Path(args.export_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    if args.export_md:
        path = Path(args.export_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown_table(rows, columns) + "\n", encoding="utf-8")
    if args.enforce_gate and gate_failed:
        raise SystemExit("Torch OSQP exceeded the configured automatic-selection gate.")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["bound", "equality", "random_spd"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 600, 1000])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maximum-slowdown", type=float, default=5.0)
    parser.add_argument("--enforce-gate", action="store_true")
    parser.add_argument("--export-csv")
    parser.add_argument("--export-md")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
