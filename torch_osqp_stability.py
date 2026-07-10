"""Generate reproducible Torch-OSQP stability evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from pygranso.private.osqpTorchAdapter import solve_osqp_torch_qp
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace

CSV_COLUMNS = (
    "test_family",
    "case_name",
    "seed",
    "device",
    "dtype",
    "builtin_status",
    "torch_status",
    "primal_residual",
    "primal_tolerance",
    "dual_residual",
    "dual_tolerance",
    "builtin_primal_residual",
    "builtin_primal_tolerance",
    "builtin_dual_residual",
    "builtin_dual_tolerance",
    "objective_builtin",
    "objective_torch",
    "relative_objective_gap",
    "bound_violation",
    "equality_violation",
    "linear_residual",
    "condition_estimate",
    "support_class",
    "pass_primal",
    "pass_dual",
    "pass_builtin_residuals",
    "pass_objective",
    "pass_status",
    "overall_status",
    "release_gate",
    "failure_reason",
)


SUPPORTED_FAMILIES = ("randomized_bounds", "randomized_equality")
STRESS_FAMILY = "conditioning_stress"


def source_tree_fingerprint():
    """Hash the maintained solver, tests, workflows, scripts, and docs."""

    root = Path(__file__).resolve().parent
    patterns = (
        "pygranso/**/*.py",
        "tests/**/*.py",
        ".github/workflows/*.yml",
        "docs/*.md",
        "scripts/*.py",
        "bench*.py",
        "torch_osqp_stability.py",
        "pyproject.toml",
        "README.md",
        ".gitignore",
    )
    paths = sorted(
        {path for pattern in patterns for path in root.glob(pattern) if path.is_file()},
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest(), len(paths)


def make_problem(family, seed, device, dtype):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n = 2 + seed % 7
    if family == STRESS_FAMILY:
        target_condition = 5e7
    elif dtype == torch.float32:
        target_condition = (2.0, 5.0, 10.0, 30.0)[seed % 4]
    else:
        target_condition = (1e2, 1e4, 1e5, 1e6)[seed % 4]
    eigenvalues = torch.logspace(
        0,
        math.log10(target_condition),
        n,
        dtype=dtype,
    )
    Q, _ = torch.linalg.qr(torch.randn((n, n), generator=generator, dtype=dtype))
    H = Q @ torch.diag(eigenvalues) @ Q.T
    f = torch.randn((n, 1), generator=generator, dtype=dtype)
    lower = -torch.ones((n, 1), dtype=dtype)
    upper = torch.ones((n, 1), dtype=dtype)
    if family in {"randomized_equality", STRESS_FAMILY}:
        A = torch.ones((1, n), dtype=dtype)
        b = torch.tensor(0.25, dtype=dtype)
    else:
        A = b = None
    values = (H, f, A, b, lower, upper)
    values = tuple(value.to(device=device) if torch.is_tensor(value) else value for value in values)
    return family, values, float(target_condition)


def solve(problem, algebra, settings):
    H, f, A, b, lower, upper = problem
    return solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        lower,
        upper,
        f.device,
        f.dtype == torch.float64,
        {"algebra": algebra, "settings": settings},
        TorchOSQPWorkspace(),
    )


def estimate_initial_kkt_condition(problem, dtype):
    """Estimate KKT conditioning on CPU without exercising a backend solver."""

    H, f, A, b, lower, upper = problem
    del f, b
    P = H.detach().cpu().to(dtype=torch.float64)
    n = int(P.shape[0])
    identity = torch.eye(n, dtype=torch.float64)
    lower_cpu = lower.detach().cpu().reshape(-1).to(dtype=torch.float64)
    upper_cpu = upper.detach().cpu().reshape(-1).to(dtype=torch.float64)
    if A is None:
        A_osqp = identity
        l_osqp = lower_cpu
        u_osqp = upper_cpu
    else:
        equality = A.detach().cpu().to(dtype=torch.float64)
        if equality.ndim == 1:
            equality = equality.reshape(1, -1)
        A_osqp = torch.cat((equality, identity), dim=0)
        rhs = torch.zeros(equality.shape[0], dtype=torch.float64)
        l_osqp = torch.cat((rhs, lower_cpu))
        u_osqp = torch.cat((rhs, upper_cpu))
    equality_mask = torch.isfinite(l_osqp) & torch.isfinite(u_osqp) & torch.isclose(
        l_osqp,
        u_osqp,
        rtol=100.0 * torch.finfo(torch.float64).eps,
        atol=100.0 * torch.finfo(torch.float64).eps,
    )
    rho_vec = torch.full((A_osqp.shape[0],), 0.1, dtype=torch.float64)
    rho_vec[equality_mask] *= 1000.0
    top = torch.cat((P + 1e-6 * identity, A_osqp.T), dim=1)
    bottom = torch.cat((A_osqp, -torch.diag(rho_vec.reciprocal())), dim=1)
    K = torch.cat((top, bottom), dim=0)
    try:
        value = torch.linalg.cond(K).item()
    except RuntimeError:
        return math.inf
    if not math.isfinite(value):
        return math.inf
    return float(value)


def evaluate(family, seed, device, dtype):
    family, problem, condition = make_problem(family, seed, device, dtype)
    H, _f, A, b, lower, upper = problem
    condition = estimate_initial_kkt_condition(problem, dtype)
    settings = {
        "return_info": True,
        "check_linear_residual": True,
        "check_condition": False,
        "eps_abs": 1e-8 if dtype == torch.float64 else 1e-5,
        "eps_rel": 1e-8 if dtype == torch.float64 else 1e-5,
    }
    torch_x, torch_info = solve(problem, "torch", settings)
    builtin_x, builtin_info = solve(problem, "builtin", settings)
    objective_scale = max(1.0, abs(builtin_info["objective"]))
    objective_gap = abs(torch_info["objective"] - builtin_info["objective"]) / objective_scale
    bounds = torch.maximum(
        torch.clamp(lower.reshape(-1) - torch_x.reshape(-1), min=0),
        torch.clamp(torch_x.reshape(-1) - upper.reshape(-1), min=0),
    )
    bound_violation = float(torch.max(bounds).item())
    equality_violation = 0.0
    if A is not None:
        equality_violation = float(torch.max(torch.abs(A @ torch_x - b)).item())
    objective_tolerance = 1e-7 if dtype == torch.float64 else 1e-4
    pass_primal = torch_info["primal_residual"] <= torch_info["eps_primal"]
    pass_dual = torch_info["dual_residual"] <= torch_info["eps_dual"]
    pass_builtin_residuals = (
        builtin_info["primal_residual"] <= builtin_info["eps_primal"]
        and builtin_info["dual_residual"] <= builtin_info["eps_dual"]
    )
    pass_objective = objective_gap <= objective_tolerance
    pass_status = bool(
        torch_info["status_compatible"] and builtin_info["status_compatible"]
    )
    passed = (
        pass_primal
        and pass_dual
        and pass_builtin_residuals
        and pass_objective
        and pass_status
    )
    supported_condition_limit = 1e8 if dtype == torch.float64 else 1e2
    support_class = (
        "stress"
        if family == STRESS_FAMILY or condition > supported_condition_limit
        else "supported"
    )
    failures = []
    if not pass_primal:
        failures.append("primal")
    if not pass_dual:
        failures.append("dual")
    if not pass_builtin_residuals:
        failures.append("builtin_residuals")
    if not pass_objective:
        failures.append("objective")
    if not pass_status:
        failures.append("status")
    return {
        "test_family": family,
        "case_name": f"{family}_seed_{seed}",
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype),
        "builtin_status": builtin_info["status"],
        "torch_status": torch_info["status"],
        "primal_residual": torch_info["primal_residual"],
        "primal_tolerance": torch_info["eps_primal"],
        "dual_residual": torch_info["dual_residual"],
        "dual_tolerance": torch_info["eps_dual"],
        "builtin_primal_residual": builtin_info["primal_residual"],
        "builtin_primal_tolerance": builtin_info["eps_primal"],
        "builtin_dual_residual": builtin_info["dual_residual"],
        "builtin_dual_tolerance": builtin_info["eps_dual"],
        "objective_builtin": builtin_info["objective"],
        "objective_torch": torch_info["objective"],
        "relative_objective_gap": objective_gap,
        "bound_violation": bound_violation,
        "equality_violation": equality_violation,
        "linear_residual": torch_info.get("maximum_linear_residual"),
        "condition_estimate": condition,
        "support_class": support_class,
        "pass_primal": pass_primal,
        "pass_dual": pass_dual,
        "pass_builtin_residuals": pass_builtin_residuals,
        "pass_objective": pass_objective,
        "pass_status": pass_status,
        "overall_status": "passed" if passed else "failed",
        "release_gate": (
            "not_applicable"
            if support_class == "stress"
            else "passed"
            if passed
            else "failed"
        ),
        "failure_reason": "|".join(failures),
    }, problem


def environment_manifest(args):
    commit = "unknown"
    dirty = None
    git_status_entries = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_output = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        git_status_entries = [line for line in status_output.splitlines() if line]
        dirty = bool(git_status_entries)
    except (OSError, subprocess.SubprocessError):
        pass
    hardware = platform.processor() or platform.machine()
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and torch.cuda.is_available():
        hardware = torch.cuda.get_device_name(requested_device)
    source_sha256, source_file_count = source_tree_fingerprint()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "git_status_entries": git_status_entries,
        "source_tree_sha256": source_sha256,
        "source_file_count": source_file_count,
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "osqp": importlib.metadata.version("osqp"),
        "device": args.device,
        "backends": ["builtin", "torch"],
        "hardware": hardware,
        "dtype": args.dtype,
        "families": list(SUPPORTED_FAMILIES),
        "seeds_per_supported_family": args.seeds,
        "supported_seeds": list(range(args.seeds)),
        "stress_seeds": list(range(args.stress_seeds)),
        "time_limit_seconds": args.time_limit_seconds,
        "supported_condition_limit": 1e8 if args.dtype == "float64" else 1e2,
        "stress_condition_limit": 1e10,
        "condition_estimate_method": "cpu_dense_initial_kkt",
        "kkt_dimension_limit": 2400,
        "settings": {
            "rho": 0.1,
            "sigma": 1e-6,
            "alpha": 1.6,
            "scaling": 10,
            "adaptive_rho": True,
            "polishing": True,
        },
    }


def write_summary(path, rows, manifest):
    families = sorted({row["test_family"] for row in rows})
    completed_cases = manifest.get("completed_cases", len(rows))
    planned_cases = manifest.get("planned_cases", len(rows))
    lines = [
        "# Torch OSQP Stability Summary",
        "",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Device: `{manifest['device']}`",
        f"- Dtype: `{manifest['dtype']}`",
        f"- Cases: {completed_cases}/{planned_cases}",
        f"- Timed out: {manifest.get('timed_out', False)}",
        f"- Passed: {sum(row['overall_status'] == 'passed' for row in rows)}",
        f"- Failed: {sum(row['overall_status'] == 'failed' for row in rows)}",
        f"- Release-gate failures: {sum(row['release_gate'] == 'failed' for row in rows)}",
        "",
        "| Test family | Total | Passed | Failed |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family in families:
        family_rows = [row for row in rows if row["test_family"] == family]
        passed = sum(row["overall_status"] == "passed" for row in family_rows)
        lines.append(f"| {family} | {len(family_rows)} | {passed} | {len(family_rows)-passed} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifacts(output, rows, manifest, started, planned_cases, timed_out):
    """Write CSV, manifest, and summary for complete or partial stability runs."""

    manifest["elapsed_seconds"] = elapsed = time.perf_counter() - started
    manifest["completed_cases"] = len(rows)
    manifest["planned_cases"] = planned_cases
    manifest["timed_out"] = bool(timed_out)
    manifest["partial_results"] = bool(timed_out or len(rows) < planned_cases)
    with (output / "torch_osqp_stability_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (output / "torch_osqp_stability_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    write_summary(output / "torch_osqp_stability_summary.md", rows, manifest)
    failures = [row for row in rows if row["release_gate"] == "failed"]
    observations = [row for row in rows if row["overall_status"] == "failed"]
    print(
        f"Stability cases: {len(rows)}/{planned_cases}, "
        f"numerical failures: {len(observations)}, "
        f"release-gate failures: {len(failures)}, elapsed: {elapsed:.2f}s"
    )
    return elapsed, failures


def run(args):
    started = time.perf_counter()
    # Capture source provenance before creating output files so the manifest's
    # dirty flag describes the solver worktree, not its own generated artifacts.
    manifest = environment_manifest(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reproduction = output / "failures"
    reproduction.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    rows = []
    cases = [
        (family, seed)
        for family in SUPPORTED_FAMILIES
        for seed in range(args.seeds)
    ]
    cases.extend((STRESS_FAMILY, seed) for seed in range(args.stress_seeds))
    planned_cases = len(cases)
    for family, seed in cases:
        try:
            row, problem = evaluate(family, seed, device, dtype)
        except Exception as exc:
            family, problem, condition = make_problem(family, seed, device, dtype)
            support_class = "stress" if family == STRESS_FAMILY else "supported"
            row = {column: None for column in CSV_COLUMNS}
            row.update(
                {
                    "test_family": family,
                    "case_name": f"{family}_seed_{seed}",
                    "seed": seed,
                    "device": str(device),
                    "dtype": str(dtype),
                    "condition_estimate": condition,
                    "support_class": support_class,
                    "overall_status": "failed",
                    "release_gate": (
                        "not_applicable" if support_class == "stress" else "failed"
                    ),
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
        rows.append(row)
        if row["overall_status"] != "passed":
            torch.save(
                {
                    "family": family,
                    "seed": seed,
                    "problem": tuple(
                        value.detach().cpu() if torch.is_tensor(value) else value
                        for value in problem
                    ),
                    "result": row,
                },
                reproduction / f"{family}_seed_{seed}.pt",
            )
        elapsed = time.perf_counter() - started
        if elapsed > args.time_limit_seconds:
            manifest["timeout_after_case"] = {
                "completed_cases": len(rows),
                "test_family": family,
                "seed": seed,
            }
            elapsed, _failures = write_artifacts(
                output,
                rows,
                manifest,
                started,
                planned_cases,
                timed_out=True,
            )
            raise SystemExit(
                f"Stability bucket exceeded {args.time_limit_seconds}s after "
                f"{len(rows)}/{planned_cases} cases: {elapsed:.2f}s."
            )
    elapsed, failures = write_artifacts(
        output,
        rows,
        manifest,
        started,
        planned_cases,
        timed_out=False,
    )
    if elapsed > args.time_limit_seconds:
        raise SystemExit(
            f"Stability bucket exceeded {args.time_limit_seconds}s: {elapsed:.2f}s."
        )
    if failures:
        raise SystemExit(1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/torch_osqp_stability")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument(
        "--seeds",
        type=int,
        default=100,
        help="Fixed seeds per supported family/backend bucket.",
    )
    parser.add_argument("--stress-seeds", type=int, default=100)
    parser.add_argument("--time-limit-seconds", type=float, default=7200.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
