"""Backend policy and canonicalization for PyGRANSO OSQP subproblems."""

from __future__ import annotations

import importlib
import math
import warnings
from numbers import Integral, Number, Real

import numpy as np
import torch
from scipy import sparse

from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchOSQP import (
    _osqp_infinity,
    _restore_workspace,
    _snapshot_workspace,
    solve_torch_osqp_direct,
)

MAX_SUPPORTED_KKT_DIM = 2400
MAX_AUTO_ESTIMATED_MEMORY_MB = 512.0
PROMOTED_ACCELERATOR_BACKENDS = {
    "cuda": False,
    "rocm": False,
    "mps": False,
}
LEGACY_TORCH_SETTINGS = {
    "linear_solver",
    "cg_rtol",
    "cg_atol",
    "cg_max_iter",
    "cg_check_interval",
    "cg_fixed_iters",
    "torch_compile_admm",
    "cuda_graph",
    "cuda_event_timing",
    "linear_solver_auto_min_kkt_dim",
    "linear_solver_auto_sparse_min_kkt_dim",
    "linear_solver_auto_max_density",
    "linear_solver_auto_dense_memory_limit_mb",
}

DEFAULT_OSQP_SETTINGS = {
    "use_sparse": False,
    "rho": 0.1,
    "sigma": 1e-6,
    "alpha": 1.6,
    "max_iter": 4000,
    "time_limit": 1e10,
    "eps_abs": 1e-8,
    "eps_rel": 1e-8,
    "eps_prim_inf": 1e-4,
    "eps_dual_inf": 1e-4,
    "check_termination": 25,
    "check_dualgap": True,
    "scaled_termination": False,
    "scaling": 10,
    "adaptive_rho": True,
    "rho_update_interval": 50,
    "adaptive_rho_fraction": 0.4,
    "rho_update_tolerance": 5.0,
    "rho_update_safeguard": False,
    "warm_start": True,
    "polishing": False,
    "polish_delta": 1e-6,
    "polish_refine_iter": 3,
    "check_linear_residual": False,
    "check_convexity": False,
    "check_condition": False,
    "symmetry_tolerance_multiplier": 100.0,
    "return_state": False,
    "return_info": False,
    "verbose": False,
}
DEFAULT_TORCH_OSQP_SETTINGS = dict(DEFAULT_OSQP_SETTINGS)


def solve_osqp_torch_qp(
    H,
    f,
    A,
    b,
    LB,
    UB,
    torch_device,
    double_precision,
    options=None,
    workspace: TorchOSQPWorkspace | None = None,
):
    """Solve PyGRANSO's QP form and return a Torch column vector."""

    torch_dtype = torch.float64 if double_precision else torch.float32
    opts = _normalize_options(options, torch_dtype)
    workspace = workspace or TorchOSQPWorkspace()
    target_device = _canonical_device(torch_device)
    _validate_torch_input_compatibility(H, f, A, b, LB, UB, expected_dtype=torch_dtype)
    if (A is None) != (b is None):
        raise ValueError("A and b must either both be provided or both be None.")
    settings = opts["settings"]
    selection = _select_backend(
        opts["algebra"],
        target_device,
        torch_dtype,
        H,
        f,
        A,
        b,
    )

    if selection["backend"] == "builtin":
        workspace_snapshot = _snapshot_workspace(workspace)
        mutation_state = {"cached_problem": False}
        backend_changed = workspace.ensure_backend("builtin")
        try:
            result_device = _builtin_result_device(
                target_device,
                torch_dtype,
                selection["selection_reason"],
            )
            solution, info = _solve_builtin_osqp_path(
                H,
                f,
                A,
                b,
                LB,
                UB,
                result_device,
                target_device,
                torch_dtype,
                settings,
                workspace,
                mutation_state,
            )
        except BaseException:
            _restore_workspace(workspace, workspace_snapshot)
            if mutation_state["cached_problem"]:
                workspace.builtin_cache = None
            raise
        selection["workspace_invalidated_for_backend_change"] = backend_changed
        info.update(selection)
        _raise_if_builtin_interrupted(
            info,
            workspace,
            workspace_snapshot,
            mutation_state,
        )
        workspace.last_info = dict(info)
        return (solution, info) if settings["return_info"] else solution

    torch_failure = None
    workspace_snapshot = _snapshot_workspace(workspace)
    backend_changed = workspace.ensure_backend("torch")
    try:
        solution, torch_info = _solve_torch_osqp_path(
            H,
            f,
            A,
            b,
            LB,
            UB,
            target_device,
            torch_dtype,
            settings,
            workspace,
            allow_device_move=opts["algebra"] == "auto",
        )
        selection["workspace_invalidated_for_backend_change"] = backend_changed
        torch_info.update(selection)
        if torch_info.get("status_compatible", False):
            workspace.last_info = dict(torch_info)
            return (solution, torch_info) if settings["return_info"] else solution
        torch_failure = {
            "trigger": "unsolved_status",
            "status": torch_info.get("status"),
            "message": "Torch OSQP did not return a solved-compatible status.",
            "torch_info": torch_info,
        }
        _restore_workspace(workspace, workspace_snapshot)
    except BaseException as exc:
        _restore_workspace(workspace, workspace_snapshot)
        if not isinstance(exc, Exception):
            raise
        if opts["algebra"] != "auto":
            raise
        torch_failure = {
            "trigger": "exception",
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }

    if opts["algebra"] != "auto":
        workspace.last_info = dict(torch_info)
        raise RuntimeError(
            "Explicit Torch OSQP did not produce a solved-compatible result: "
            f"{torch_info.get('status', 'unknown')}."
        )

    warnings.warn(
        "Automatic Torch OSQP did not produce a solved-compatible result; "
        f"falling back to builtin CPU OSQP ({torch_failure['message']}).",
        RuntimeWarning,
        stacklevel=2,
    )
    workspace_snapshot = _snapshot_workspace(workspace)
    mutation_state = {"cached_problem": False}
    backend_changed = workspace.ensure_backend("builtin")
    try:
        solution, builtin_info = _solve_builtin_osqp_path(
            H,
            f,
            A,
            b,
            LB,
            UB,
            _builtin_result_device(
                target_device,
                torch_dtype,
                "torch_failed_or_unsolved",
            ),
            target_device,
            torch_dtype,
            settings,
            workspace,
            mutation_state,
        )
    except BaseException:
        _restore_workspace(workspace, workspace_snapshot)
        if mutation_state["cached_problem"]:
            workspace.builtin_cache = None
        raise
    builtin_info["fallback"] = {
        "occurred": True,
        "requested_backend": "auto",
        "selected_backend": "torch",
        "fallback_backend": "builtin",
        "device_transfer": target_device.type != "cpu",
        "workspace_invalidated_for_backend_change": backend_changed,
        **torch_failure,
    }
    builtin_info.update(
        {
            "backend": "builtin",
            "selection_reason": "torch_failed_or_unsolved",
        }
    )
    _raise_if_builtin_interrupted(
        builtin_info,
        workspace,
        workspace_snapshot,
        mutation_state,
    )
    workspace.last_info = dict(builtin_info)
    return (solution, builtin_info) if settings["return_info"] else solution


def _raise_if_builtin_interrupted(
    info,
    workspace,
    workspace_snapshot,
    mutation_state,
):
    if info.get("status") != "interrupted":
        return
    _restore_workspace(workspace, workspace_snapshot)
    if mutation_state["cached_problem"]:
        workspace.builtin_cache = None
    workspace.last_info = dict(info)
    raise KeyboardInterrupt("Builtin OSQP solve was interrupted by the user.")


def _select_backend(algebra, target_device, dtype, H, f, A, b):
    kkt_dim, memory_components = _dense_kkt_memory_estimate(H, f, A, b, dtype)
    memory_mb = sum(memory_components.values())
    metadata = {
        "requested_backend": algebra,
        "estimated_kkt_dim": kkt_dim,
        "estimated_dense_working_memory_mb": memory_mb,
        "estimated_dense_memory_components_mb": memory_components,
    }
    if algebra == "builtin":
        return {
            **metadata,
            "backend": "builtin",
            "selection_reason": "explicit",
            "fallback": {"occurred": False},
        }
    if algebra == "torch":
        if kkt_dim > MAX_SUPPORTED_KKT_DIM or memory_mb > _memory_limit_mb(
            target_device
        ):
            warnings.warn(
                "Explicit Torch OSQP exceeds the validated dense envelope "
                f"(KKT dimension {kkt_dim}, estimated {memory_mb:.1f} MiB); "
                "attempting the solve as requested.",
                RuntimeWarning,
                stacklevel=3,
            )
        return {
            **metadata,
            "backend": "torch",
            "selection_reason": "explicit",
            "fallback": {"occurred": False},
        }

    if target_device.type == "cpu":
        return {
            **metadata,
            "backend": "builtin",
            "selection_reason": "cpu_target_uses_builtin",
            "fallback": {"occurred": False},
        }
    supported, reason = _accelerator_capability(target_device, dtype)
    if not supported:
        warnings.warn(
            f"Torch OSQP is not validated for {target_device}/{dtype}: {reason}. "
            "Falling back to builtin CPU OSQP.",
            RuntimeWarning,
            stacklevel=3,
        )
        return _selection_fallback(metadata, reason, target_device)
    if kkt_dim > MAX_SUPPORTED_KKT_DIM:
        warnings.warn(
            f"Automatic Torch OSQP KKT dimension {kkt_dim} exceeds the "
            f"validated limit {MAX_SUPPORTED_KKT_DIM}; using builtin OSQP.",
            RuntimeWarning,
            stacklevel=3,
        )
        return _selection_fallback(metadata, "kkt_dimension_limit", target_device)
    if memory_mb > _memory_limit_mb(target_device):
        warnings.warn(
            f"Automatic Torch OSQP estimated memory {memory_mb:.1f} MiB "
            "exceeds the conservative preflight; using builtin OSQP.",
            RuntimeWarning,
            stacklevel=3,
        )
        return _selection_fallback(metadata, "memory_preflight", target_device)
    return {
        **metadata,
        "backend": "torch",
        "selection_reason": "validated_accelerator_target",
        "fallback": {"occurred": False},
    }


def _selection_fallback(metadata, reason, target_device):
    return {
        **metadata,
        "backend": "builtin",
        "selection_reason": reason,
        "fallback": {
            "occurred": True,
            "trigger": "selection_policy",
            "requested_backend": "auto",
            "selected_backend": "builtin",
            "fallback_backend": "builtin",
            "reason": reason,
            "device_transfer": target_device.type != "cpu",
        },
    }


def estimate_dense_kkt(H, f, A, b, dtype):
    kkt_dim, components = _dense_kkt_memory_estimate(H, f, A, b, dtype)
    return kkt_dim, float(sum(components.values()))


def _dense_kkt_memory_estimate(H, f, A, b, dtype):
    n = int(_shape_length(f))
    n_eq = 0
    if A is not None and b is not None:
        shape = tuple(A.shape) if hasattr(A, "shape") else np.asarray(A).shape
        n_eq = 1 if len(shape) == 1 else int(shape[0])
    constraint_rows = n_eq + n
    kkt_dim = n + constraint_rows
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    mib = 1024 * 1024
    element_counts = {
        "source_matrices": n * n + n_eq * n,
        "bound_identity": n * n,
        "canonical_constraint_matrix": constraint_rows * n,
        "kkt_matrix": kkt_dim * kkt_dim,
        "kkt_factorization_workspace": 2 * kkt_dim * kkt_dim,
    }
    components = {
        name: float(count * dtype_bytes / mib) for name, count in element_counts.items()
    }
    return int(kkt_dim), components


def _shape_length(value):
    if torch.is_tensor(value):
        return value.numel()
    return np.asarray(value).size


def _canonical_device(device):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _builtin_result_device(requested_device, dtype, selection_reason):
    """Choose a representable result device for a CPU builtin solve."""

    cpu_only_reasons = {
        "cuda_unavailable",
        "mps_unavailable",
        "mps_float64_unsupported",
    }
    if selection_reason in cpu_only_reasons or selection_reason.startswith(
        "unvalidated_device_"
    ):
        return torch.device("cpu")
    if requested_device.type == "mps" and dtype == torch.float64:
        warnings.warn(
            "Builtin OSQP cannot return float64 on MPS; returning the CPU "
            "float64 solution.",
            RuntimeWarning,
            stacklevel=3,
        )
        return torch.device("cpu")
    return requested_device


def _memory_limit_mb(device):
    limit = MAX_AUTO_ESTIMATED_MEMORY_MB
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            total_mb = torch.cuda.get_device_properties(device).total_memory / (
                1024 * 1024
            )
            limit = min(limit, 0.25 * total_mb)
        except (AssertionError, RuntimeError):
            pass
    return float(limit)


def _accelerator_capability(device, dtype):
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return False, "cuda_unavailable"
        backend = "rocm" if torch.version.hip is not None else "cuda"
        if not PROMOTED_ACCELERATOR_BACKENDS[backend]:
            return False, f"{backend}_not_promoted"
        return True, f"{backend}_promoted"
    if device.type == "mps":
        if dtype == torch.float64:
            return False, "mps_float64_unsupported"
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            return False, "mps_unavailable"
        if not PROMOTED_ACCELERATOR_BACKENDS["mps"]:
            return False, "mps_not_promoted"
        return True, "mps_promoted"
    return False, f"unvalidated_device_{device.type}"


def _solve_torch_osqp_path(
    H,
    f,
    A,
    b,
    LB,
    UB,
    target_device,
    dtype,
    settings,
    workspace,
    allow_device_move,
):
    sparse_to_dense = _sparse_to_dense_diagnostics(
        {"H": H, "f": f, "A": A, "b": b, "LB": LB, "UB": UB},
        target_device,
    )
    if sparse_to_dense["occurred"]:
        names = ", ".join(sparse_to_dense["inputs"])
        warnings.warn(
            "Torch OSQP's dense compatibility route will convert sparse "
            f"input(s) {names} to strided tensors on {target_device}; "
            + ("the KKT solve then uses sparse CSR." if settings["use_sparse"]
               else "no sparse solver is selected."),
            RuntimeWarning,
            stacklevel=3,
        )
    if allow_device_move and sparse_to_dense["device_transfer"]:
        names = ", ".join(sparse_to_dense["device_transfers"])
        warnings.warn(
            "Automatic Torch OSQP will move input(s) "
            f"{names} to {target_device}; the transfer is recorded in diagnostics.",
            RuntimeWarning,
            stacklevel=3,
        )
    P = _torch_tensor(H, "H", target_device, dtype, allow_device_move)
    q = _torch_tensor(f, "f", target_device, dtype, allow_device_move).reshape(-1)
    n = q.numel()
    if P.shape != (n, n):
        raise ValueError(f"H must have shape {(n, n)}, got {tuple(P.shape)}.")
    A_osqp, l_osqp, u_osqp = _build_constraints_torch(
        A,
        b,
        LB,
        UB,
        n,
        target_device,
        dtype,
        allow_device_move,
    )
    # Adapter policy, not part of the C translation: the solver core multiplies
    # IEEE-infinite bounds as C does (Inf * 0 = NaN in the duality gap).  Bounds
    # beyond the OSQP infinity sentinel are replaced by the sentinel here, so
    # PyGRANSO callers can keep passing +-inf for "no bound".
    infinity = _osqp_infinity(q)
    l_osqp = torch.clamp(l_osqp, min=-infinity)
    u_osqp = torch.clamp(u_osqp, max=infinity)
    solve_settings = dict(settings)
    solve_settings["return_info"] = True
    equality_rows = 0 if A is None or b is None else int(A_osqp.shape[0] - n)
    solve_settings["_constraint_order_signature"] = (
        "pygranso_equalities",
        equality_rows,
        "variable_bounds",
        n,
    )
    solution, info = solve_torch_osqp_direct(
        P,
        q,
        A_osqp,
        l_osqp,
        u_osqp,
        solve_settings,
        workspace,
    )
    info["sparse_to_dense"] = sparse_to_dense
    info["input_device_transfer"] = {
        "occurred": sparse_to_dense["device_transfer"],
        "inputs": sparse_to_dense["device_transfers"],
        "target_device": str(target_device),
    }
    return solution.to(device=target_device, dtype=dtype), info


def _sparse_to_dense_diagnostics(values, target_device):
    tensor_inputs = {
        name: value for name, value in values.items() if torch.is_tensor(value)
    }
    inputs = {
        name: {
            "layout": str(value.layout),
            "source_device": str(value.device),
        }
        for name, value in values.items()
        if torch.is_tensor(value) and value.layout != torch.strided
    }
    device_transfers = {
        name: {
            "source_device": str(value.device),
            "target_device": str(target_device),
        }
        for name, value in tensor_inputs.items()
        if value.device != target_device
    }
    return {
        "occurred": bool(inputs),
        "inputs": inputs,
        "conversion_device": str(target_device),
        "device_transfer": bool(device_transfers),
        "device_transfers": device_transfers,
        "method": "Tensor.to(device).to_dense" if inputs else None,
    }


def _normalize_builtin_status(raw_status):
    normalized = raw_status.strip().lower().replace(" ", "_")
    return {
        "solved": "solved",
        "solved_inaccurate": "solved_inaccurate",
        "primal_infeasible": "primal_infeasible",
        "primal_infeasible_inaccurate": "primal_infeasible_inaccurate",
        "dual_infeasible": "dual_infeasible",
        "dual_infeasible_inaccurate": "dual_infeasible_inaccurate",
        "maximum_iterations_reached": "max_iter_reached",
        "run_time_limit_reached": "time_limit_reached",
        "problem_non_convex": "non_convex",
        "interrupted_by_user": "interrupted",
        "unsolved": "unsolved",
    }.get(normalized, normalized)


def _builtin_certificate(value, device, dtype):
    if value is None:
        return None
    certificate = np.asarray(value, dtype=np.float64).reshape(-1)
    if certificate.size == 0 or not np.all(np.isfinite(certificate)):
        return None
    certificate_norm = np.linalg.norm(certificate, ord=np.inf)
    if not np.isfinite(certificate_norm) or certificate_norm <= 0.0:
        return None
    return torch.from_numpy(certificate / certificate_norm).to(
        device=device,
        dtype=dtype,
    )


def _solve_builtin_osqp_path(
    H,
    f,
    A,
    b,
    LB,
    UB,
    result_device,
    requested_device,
    dtype,
    settings,
    workspace,
    mutation_state,
):
    osqp = _import_osqp()
    H_np = _to_numpy(H).astype(np.float64, copy=False)
    f_np = _to_numpy(f).reshape(-1).astype(np.float64, copy=False)
    LB_np = _to_numpy(LB).reshape(-1, 1).astype(np.float64, copy=False)
    UB_np = _to_numpy(UB).reshape(-1, 1).astype(np.float64, copy=False)
    n = f_np.size
    if H_np.shape != (n, n):
        raise ValueError(f"H must have shape {(n, n)}, got {H_np.shape}.")
    if LB_np.shape != (n, 1) or UB_np.shape != (n, 1):
        raise ValueError("LB and UB must be column vectors with len(f) rows.")
    if not np.all(np.isfinite(H_np)) or not np.all(np.isfinite(f_np)):
        raise ValueError("H and f must contain finite values.")
    asymmetry = np.linalg.norm(H_np - H_np.T, ord=np.inf)
    scale = max(1.0, np.linalg.norm(H_np, ord=np.inf))
    input_epsilon = np.finfo(np.float64 if dtype == torch.float64 else np.float32).eps
    tolerance = settings["symmetry_tolerance_multiplier"] * input_epsilon * scale
    if asymmetry > tolerance:
        raise ValueError("H is materially asymmetric for the builtin OSQP route.")
    H_np = 0.5 * (H_np + H_np.T)
    P = sparse.triu(sparse.csc_matrix(H_np), format="csc")
    A_osqp, l_osqp, u_osqp = _build_constraints_numpy(A, b, LB_np, UB_np, n)
    if np.any(np.isnan(l_osqp)) or np.any(np.isnan(u_osqp)) or np.any(l_osqp > u_osqp):
        raise ValueError("Constraint bounds are invalid.")

    builtin_settings = _builtin_settings(settings)
    settings_signature = tuple(sorted(builtin_settings.items()))
    cache = workspace.builtin_cache
    cache_hit = bool(
        cache
        and _same_csc_structure(cache["P"], P)
        and _same_csc_structure(cache["A"], A_osqp)
        and cache.get("settings_signature") == settings_signature
    )
    if cache_hit:
        mutation_state["cached_problem"] = True
        problem = cache["problem"]
        updates = {"q": f_np, "l": l_osqp, "u": u_osqp}
        if not np.array_equal(cache["P"].data, P.data):
            updates["Px"] = P.data
        if not np.array_equal(cache["A"].data, A_osqp.data):
            updates["Ax"] = A_osqp.data
        problem.update(**updates)
        previous = cache.get("result")
        warm_start_x = cache.get("warm_start_x")
        warm_start_y = cache.get("warm_start_y")
        if warm_start_x is None or warm_start_y is None:
            warm_start_x = None if previous is None else previous.x
            warm_start_y = None if previous is None else previous.y
        if (
            settings["warm_start"]
            and warm_start_x is not None
            and warm_start_y is not None
        ):
            problem.warm_start(x=warm_start_x, y=warm_start_y)
        workspace.builtin_stats["updates"] += 1
    else:
        problem = osqp.OSQP(algebra="builtin")
        problem.setup(P, f_np, A_osqp, l_osqp, u_osqp, **builtin_settings)
        workspace.builtin_stats["setups"] += 1
        if cache is not None:
            workspace.builtin_stats["rebuilds"] += 1
    result = problem.solve(raise_error=False)
    candidate_cache = {
        "problem": problem,
        "P": P,
        "A": A_osqp,
        "result": result,
        "settings_signature": settings_signature,
    }
    workspace.builtin_stats["last_cache_hit"] = cache_hit

    osqp_info = getattr(result, "info", None)
    raw_status = str(getattr(osqp_info, "status", "unknown"))
    status = _normalize_builtin_status(raw_status)
    status_val = int(getattr(osqp_info, "status_val", 11))
    primal_infeasible = status.startswith("primal_infeasible")
    dual_infeasible = status.startswith("dual_infeasible")
    status_compatible = status in {"solved", "solved_inaccurate"}
    workspace.builtin_cache = candidate_cache if status_compatible else None
    primal_certificate = (
        _builtin_certificate(getattr(result, "prim_inf_cert", None), result_device, dtype)
        if primal_infeasible
        else None
    )
    dual_certificate = (
        _builtin_certificate(getattr(result, "dual_inf_cert", None), result_device, dtype)
        if dual_infeasible
        else None
    )
    polish_status = getattr(osqp_info, "status_polish", None)
    infeasible = primal_infeasible or dual_infeasible
    no_iterate = infeasible or status == "non_convex"
    if primal_infeasible and (
        primal_certificate is None or primal_certificate.numel() != A_osqp.shape[0]
    ):
        raise RuntimeError(
            "Builtin OSQP reported primal infeasibility without a valid certificate."
        )
    if dual_infeasible and (
        dual_certificate is None or dual_certificate.numel() != n
    ):
        raise RuntimeError(
            "Builtin OSQP reported dual infeasibility without a valid certificate."
        )

    if no_iterate:
        primal = np.full((n, 1), np.nan, dtype=np.float64)
        metrics = {
            "primal_residual": float(getattr(osqp_info, "prim_res", np.nan)),
            "dual_residual": float(getattr(osqp_info, "dual_res", np.nan)),
            "eps_primal": float("nan"),
            "eps_dual": float("nan"),
            "objective": (
                float("inf")
                if primal_infeasible
                else -float("inf")
                if dual_infeasible
                else float("nan")
            ),
        }
    else:
        primal = getattr(result, "x", None)
        if primal is None or np.asarray(primal).size == 0:
            raise RuntimeError("Builtin OSQP did not return a primal solution.")
        primal = np.asarray(primal).reshape(n, 1)
        if not np.all(np.isfinite(primal)):
            raise RuntimeError("Builtin OSQP returned a non-finite solution.")
        dual_solution = getattr(result, "y", None)
        if dual_solution is None:
            raise RuntimeError("Builtin OSQP did not return a dual solution.")
        dual_solution = np.asarray(dual_solution, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(dual_solution)):
            raise RuntimeError("Builtin OSQP returned a non-finite dual solution.")
        x_vector = primal.reshape(-1)
        ax = np.asarray(A_osqp @ x_vector).reshape(-1)
        z = np.maximum(np.minimum(ax, u_osqp), l_osqp)
        metrics = _builtin_common_metrics(
            H_np, f_np, A_osqp, x_vector, z, dual_solution, settings
        )
    residual_contract_satisfied = not status_compatible or (
        metrics["primal_residual"] <= metrics["eps_primal"]
        and metrics["dual_residual"] <= metrics["eps_dual"]
    )
    info = {
        "status": status,
        "status_val": status_val,
        "status_compatible": status_compatible,
        "backend": "builtin",
        "rho_update_safeguard": False,
        "status_polish": polish_status,
        "objective": (
            float("inf")
            if primal_infeasible
            else -float("inf")
            if dual_infeasible
            else metrics["objective"]
        ),
        "primal_infeasibility_certificate": primal_certificate,
        "dual_infeasibility_certificate": dual_certificate,
        "prim_inf_cert": primal_certificate,
        "dual_inf_cert": dual_certificate,
        "primal_residual": metrics["primal_residual"],
        "dual_residual": metrics["dual_residual"],
        "eps_primal": metrics["eps_primal"],
        "eps_dual": metrics["eps_dual"],
        "solver_objective": float(getattr(osqp_info, "obj_val", np.nan)),
        "solver_primal_residual": float(getattr(osqp_info, "prim_res", np.nan)),
        "solver_dual_residual": float(getattr(osqp_info, "dual_res", np.nan)),
        "adapter_residual_contract_satisfied": residual_contract_satisfied,
        "admm_iterations": int(getattr(osqp_info, "iter", 0)),
        "device": str(result_device),
        "requested_device": str(requested_device),
        "dtype": str(dtype),
        "device_transfer": requested_device.type != "cpu",
        "builtin_workspace": dict(workspace.builtin_stats),
        "fallback": {"occurred": False},
        "polishing": bool(settings["polishing"]),
        "polishing_status": None if polish_status is None else int(polish_status),
        "polishing_fallback": None,
    }
    return torch.from_numpy(primal).to(device=result_device, dtype=dtype), info


def _builtin_common_metrics(H, f, A, x, z, y, settings):
    ax = np.asarray(A @ x).reshape(-1)
    px = H @ x
    aty = np.asarray(A.T @ y).reshape(-1)
    return {
        "primal_residual": float(np.linalg.norm(ax - z, ord=np.inf)),
        "dual_residual": float(np.linalg.norm(px + f + aty, ord=np.inf)),
        "eps_primal": float(
            settings["eps_abs"]
            + settings["eps_rel"]
            * max(np.linalg.norm(ax, ord=np.inf), np.linalg.norm(z, ord=np.inf))
        ),
        "eps_dual": float(
            settings["eps_abs"]
            + settings["eps_rel"]
            * max(
                np.linalg.norm(px, ord=np.inf),
                np.linalg.norm(aty, ord=np.inf),
                np.linalg.norm(f, ord=np.inf),
            )
        ),
        "objective": float(0.5 * x @ H @ x + f @ x),
    }


def _normalize_options(options, dtype):
    if options is None:
        options = {}
    elif not isinstance(options, dict):
        raise ValueError("osqp options must be a dict.")
    algebra = options.get("algebra", "torch")
    if algebra not in {"auto", "builtin", "torch"}:
        raise ValueError("osqp_algebra must be 'auto', 'builtin', or 'torch'.")
    user_settings = options.get("settings")
    if user_settings is None:
        user_settings = {}
    elif not isinstance(user_settings, dict):
        raise ValueError("osqp settings must be a dict.")
    legacy = sorted(LEGACY_TORCH_SETTINGS.intersection(user_settings))
    if legacy:
        warnings.warn(
            "Sparse-CG and CUDA Graph Torch OSQP settings are deprecated and "
            "archived. Remove them and select osqp_algebra='torch'.",
            FutureWarning,
            stacklevel=3,
        )
        raise ValueError(
            "Archived Torch OSQP settings are no longer executable: "
            + ", ".join(legacy)
            + ". Remove these keys; the Torch route now uses one validated direct backend."
        )
    settings = _default_settings(dtype)
    for key, value in user_settings.items():
        normalized = "polishing" if key == "polish" else key
        if key == "polish":
            warnings.warn(
                "OSQP setting 'polish' is deprecated; use 'polishing'.",
                FutureWarning,
                stacklevel=3,
            )
        if normalized not in settings:
            raise ValueError(f"Unsupported common OSQP setting {key!r}.")
        settings[normalized] = value
    _validate_settings(settings)
    if settings["use_sparse"] and algebra != "torch":
        raise ValueError("use_sparse=True requires osqp_algebra='torch'.")
    return {"algebra": algebra, "settings": settings}


def _default_settings(dtype):
    settings = dict(DEFAULT_OSQP_SETTINGS)
    tolerance = 1e-8 if dtype == torch.float64 else 1e-5
    settings["eps_abs"] = tolerance
    settings["eps_rel"] = tolerance
    settings["check_dualgap"] = dtype == torch.float64
    return settings


def _validate_settings(settings):
    settings["rho"] = _positive_float(settings["rho"], "rho")
    settings["sigma"] = _positive_float(settings["sigma"], "sigma")
    settings["alpha"] = _positive_float(settings["alpha"], "alpha")
    if settings["alpha"] >= 2:
        raise ValueError("alpha must be in (0, 2).")
    settings["max_iter"] = _positive_int(settings["max_iter"], "max_iter")
    settings["time_limit"] = _positive_float(settings["time_limit"], "time_limit")
    settings["eps_abs"] = _nonnegative_float(settings["eps_abs"], "eps_abs")
    settings["eps_rel"] = _nonnegative_float(settings["eps_rel"], "eps_rel")
    if settings["eps_abs"] == 0.0 and settings["eps_rel"] == 0.0:
        raise ValueError("At least one of eps_abs and eps_rel must be positive.")
    settings["eps_prim_inf"] = _positive_float(settings["eps_prim_inf"], "eps_prim_inf")
    settings["eps_dual_inf"] = _positive_float(settings["eps_dual_inf"], "eps_dual_inf")
    settings["check_termination"] = _nonnegative_int(
        settings["check_termination"], "check_termination"
    )
    settings["scaling"] = _nonnegative_int(settings["scaling"], "scaling")
    settings["adaptive_rho"] = _adaptive_rho_mode(settings["adaptive_rho"])
    settings["rho_update_interval"] = _nonnegative_int(
        settings["rho_update_interval"], "rho_update_interval"
    )
    settings["adaptive_rho_fraction"] = _positive_float(
        settings["adaptive_rho_fraction"], "adaptive_rho_fraction"
    )
    settings["rho_update_tolerance"] = _positive_float(
        settings["rho_update_tolerance"], "rho_update_tolerance"
    )
    if settings["rho_update_tolerance"] < 1.0:
        raise ValueError("rho_update_tolerance must be at least 1.0.")
    settings["warm_start"] = _bool(settings["warm_start"], "warm_start")
    settings["polishing"] = _bool(settings["polishing"], "polishing")
    settings["polish_delta"] = _positive_float(settings["polish_delta"], "polish_delta")
    settings["polish_refine_iter"] = _nonnegative_int(
        settings["polish_refine_iter"], "polish_refine_iter"
    )
    settings["symmetry_tolerance_multiplier"] = _positive_float(
        settings["symmetry_tolerance_multiplier"], "symmetry_tolerance_multiplier"
    )
    for key in (
        "use_sparse",
        "rho_update_safeguard",
        "check_linear_residual",
        "check_dualgap",
        "scaled_termination",
        "check_convexity",
        "check_condition",
        "return_state",
        "return_info",
        "verbose",
    ):
        settings[key] = _bool(settings[key], key)
    if settings["rho_update_safeguard"]:
        raise ValueError("The pinned C rho update requires rho_update_safeguard=False.")


def _builtin_settings(settings):
    return {
        "rho": settings["rho"],
        "sigma": settings["sigma"],
        "alpha": settings["alpha"],
        "max_iter": settings["max_iter"],
        "time_limit": settings["time_limit"],
        "eps_abs": settings["eps_abs"],
        "eps_rel": settings["eps_rel"],
        "eps_prim_inf": settings["eps_prim_inf"],
        "eps_dual_inf": settings["eps_dual_inf"],
        "check_termination": settings["check_termination"],
        "check_dualgap": settings["check_dualgap"],
        "scaled_termination": settings["scaled_termination"],
        "scaling": settings["scaling"],
        "adaptive_rho": settings["adaptive_rho"],
        "adaptive_rho_interval": settings["rho_update_interval"],
        "adaptive_rho_fraction": settings["adaptive_rho_fraction"],
        "adaptive_rho_tolerance": settings["rho_update_tolerance"],
        "warm_starting": settings["warm_start"],
        "polishing": settings["polishing"],
        "delta": settings["polish_delta"],
        "polish_refine_iter": settings["polish_refine_iter"],
        "verbose": settings["verbose"],
    }


def _build_constraints_torch(A, b, LB, UB, n, device, dtype, allow_device_move):
    lower = _torch_tensor(LB, "LB", device, dtype, allow_device_move).reshape(-1)
    upper = _torch_tensor(UB, "UB", device, dtype, allow_device_move).reshape(-1)
    if lower.numel() != n or upper.numel() != n:
        raise ValueError("LB and UB must have len(f) entries.")
    identity = torch.eye(n, device=device, dtype=dtype)
    if A is None or b is None:
        return identity, lower, upper
    equality = _torch_tensor(A, "A", device, dtype, allow_device_move)
    if equality.ndim == 1:
        equality = equality.reshape(1, -1)
    if equality.ndim != 2 or equality.shape[1] != n:
        raise ValueError(f"A must have {n} columns.")
    rhs = _torch_rhs(b, device, dtype, allow_device_move).reshape(-1)
    if rhs.numel() == 1 and equality.shape[0] != 1:
        rhs = rhs.expand(equality.shape[0])
    if rhs.numel() != equality.shape[0]:
        raise ValueError("b must be scalar or have one entry per row of A.")
    return (
        torch.cat((equality, identity), dim=0),
        torch.cat((rhs, lower), dim=0),
        torch.cat((rhs, upper), dim=0),
    )


def _build_constraints_numpy(A, b, LB, UB, n):
    identity = sparse.eye(n, format="csc")
    if A is None or b is None:
        return identity, LB.reshape(-1), UB.reshape(-1)
    equality = _to_numpy(A)
    if equality.ndim == 1:
        equality = equality.reshape(1, -1)
    if equality.ndim != 2 or equality.shape[1] != n:
        raise ValueError(f"A must have {n} columns.")
    rhs = _to_numpy(b).reshape(-1)
    if rhs.size == 1 and equality.shape[0] != 1:
        rhs = np.full(equality.shape[0], float(rhs[0]))
    if rhs.size != equality.shape[0]:
        raise ValueError("b must be scalar or have one entry per row of A.")
    return (
        sparse.vstack((sparse.csc_matrix(equality), identity), format="csc"),
        np.concatenate((rhs, LB.reshape(-1))),
        np.concatenate((rhs, UB.reshape(-1))),
    )


def _torch_tensor(value, name, device, dtype, allow_device_move):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a Torch tensor for the Torch OSQP route.")
    tensor = value.detach()
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must use {dtype}, got {tensor.dtype}.")
    if tensor.device != device:
        if not allow_device_move:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}.")
        tensor = tensor.to(device=device)
    if tensor.layout != torch.strided:
        tensor = tensor.to_dense()
    return tensor


def _torch_rhs(value, device, dtype, allow_device_move):
    if torch.is_tensor(value):
        return _torch_tensor(value, "b", device, dtype, allow_device_move)
    if isinstance(value, Number):
        return torch.tensor(value, device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _to_numpy(value):
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.layout != torch.strided:
            tensor = tensor.to_dense()
        return tensor.numpy()
    return np.asarray(value)


def _validate_torch_input_compatibility(
    H,
    f,
    A,
    b,
    LB,
    UB,
    *,
    expected_dtype,
):
    tensors = [
        (name, value)
        for name, value in (
            ("H", H),
            ("f", f),
            ("A", A),
            ("b", b),
            ("LB", LB),
            ("UB", UB),
        )
        if torch.is_tensor(value)
    ]
    if not tensors:
        return
    reference_device = tensors[0][1].device
    for name, tensor in tensors:
        if tensor.device != reference_device:
            raise ValueError(
                "Torch QP inputs must share one device; "
                f"{name} is on {tensor.device}, expected {reference_device}."
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(
                "Torch QP inputs must match the requested precision; "
                f"{name} uses {tensor.dtype}, expected {expected_dtype}."
            )


def _same_csc_structure(left, right):
    return (
        left.shape == right.shape
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
    )


def _import_osqp():
    try:
        return importlib.import_module("osqp")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install the OSQP Python package for builtin solves."
        ) from exc


def reset_builtin_osqp_workspace(workspace=None):
    if workspace is not None:
        workspace.reset_builtin()


def get_builtin_osqp_workspace_stats(workspace=None):
    if workspace is None:
        return {"setups": 0, "updates": 0, "rebuilds": 0, "last_cache_hit": False}
    return dict(workspace.builtin_stats)


def _positive_float(value, name):
    value = _float(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _nonnegative_float(value, name):
    value = _float(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _float(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _bool(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool.")
    return value


def _adaptive_rho_mode(value):
    if isinstance(value, bool):
        return value
    if not isinstance(value, Integral) or int(value) not in {0, 1, 2, 3}:
        raise ValueError("adaptive_rho must be a bool or an integer mode in {0,1,2,3}.")
    return int(value)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)
