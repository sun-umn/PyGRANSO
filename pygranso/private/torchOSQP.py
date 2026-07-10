"""Dense Torch reference implementation of OSQP's ADMM equations."""

from __future__ import annotations

import time

import torch

from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchLinearSolve import DenseLUSolver, TorchLinearSolveError


def build_kkt_matrix(P, A, sigma, rho_vec):
    n = P.shape[0]
    identity = torch.eye(n, device=P.device, dtype=P.dtype)
    top = torch.cat((P + float(sigma) * identity, A.T), dim=1)
    bottom = torch.cat((A, -torch.diag(rho_vec.reciprocal())), dim=1)
    return torch.cat((top, bottom), dim=0)


def build_kkt_rhs(x, z, y, q, sigma, rho_vec):
    return torch.cat((float(sigma) * x - q, z - y / rho_vec))


def recover_z_tilde(z, nu, y, rho_vec):
    return z + (nu - y) / rho_vec


def admm_vector_update(x_tilde, x, z_tilde, z, y, rho_vec, l, u, alpha):
    x_next = float(alpha) * x_tilde + (1.0 - float(alpha)) * x
    z_relaxed = float(alpha) * z_tilde + (1.0 - float(alpha)) * z
    z_next = torch.maximum(torch.minimum(z_relaxed + y / rho_vec, u), l)
    y_next = y + rho_vec * (z_relaxed - z_next)
    return x_next, z_next, y_next


def solve_torch_osqp_direct(P, q, A, l, u, settings, workspace=None):
    """Solve ``min 0.5*x'Px + q'x`` subject to ``l <= Ax <= u``."""

    workspace = workspace or TorchOSQPWorkspace()
    with torch.no_grad():
        P, q, A, l, u = _validate_qp(P, q, A, l, u, settings)
        _prepare_workspace(
            workspace,
            P,
            A,
            settings.get("_constraint_order_signature"),
        )
        scaling = _scaling_for_problem(workspace, P, q, A, settings)
        P_s, q_s, A_s, l_s, u_s = _scale_problem(P, q, A, l, u, scaling)

        n = q.numel()
        m = l.numel()
        x, z, y = _initial_scaled_state(
            workspace,
            n,
            m,
            q.device,
            q.dtype,
            scaling,
            bool(settings.get("warm_start", True)),
        )
        initial_rho = (
            workspace.rho_bar
            if (
                bool(settings.get("warm_start", True))
                and workspace.rho_bar is not None
                and workspace.rho_setting == float(settings["rho"])
            )
            else float(settings["rho"])
        )
        rho_bar = torch.as_tensor(initial_rho, device=q.device, dtype=q.dtype)
        equality = _equality_mask(l_s, u_s)
        rho_vec = _rho_vector(rho_bar, equality)
        sigma = float(settings["sigma"])
        alpha = float(settings["alpha"])
        max_iter = int(settings["max_iter"])
        check_termination = int(settings["check_termination"])
        adaptive_rho = bool(settings.get("adaptive_rho", True))
        rho_interval = int(settings.get("rho_update_interval", 50))
        rho_tolerance = float(settings.get("rho_update_tolerance", 5.0))
        check_linear_residual = bool(settings.get("check_linear_residual", False))

        solver = workspace.linear_solver
        factors_before = solver.factorization_count
        solves_before = solver.solve_count
        K = build_kkt_matrix(P_s, A_s, sigma, rho_vec)
        factorization_reused = not solver.factorize_if_needed(K)
        maximum_linear_residual = None
        latest_linear_diagnostics = None
        status = "max_iter_reached"
        rho_updates = 0
        last_residuals = None

        for iteration in range(1, max_iter + 1):
            rhs = build_kkt_rhs(x, z, y, q_s, sigma, rho_vec)
            solution, linear_diagnostics = solver.solve(
                rhs,
                calculate_residual=check_linear_residual,
            )
            latest_linear_diagnostics = linear_diagnostics
            observed = linear_diagnostics.get("relative_linear_residual")
            if observed is not None:
                maximum_linear_residual = (
                    observed
                    if maximum_linear_residual is None
                    else max(maximum_linear_residual, observed)
                )

            x_tilde = solution[:n]
            nu = solution[n:]
            z_tilde = recover_z_tilde(z, nu, y, rho_vec)
            x, z, y = admm_vector_update(
                x_tilde,
                x,
                z_tilde,
                z,
                y,
                rho_vec,
                l_s,
                u_s,
                alpha,
            )

            should_check = iteration % check_termination == 0
            should_update_rho = adaptive_rho and iteration % rho_interval == 0
            if should_check or should_update_rho or iteration == max_iter:
                x_o, z_o, y_o = _unscale_state(x, z, y, scaling)
                last_residuals = _residuals(
                    P,
                    q,
                    A,
                    x_o,
                    z_o,
                    y_o,
                    float(settings["eps_abs"]),
                    float(settings["eps_rel"]),
                )
                primal, dual, eps_primal, eps_dual = last_residuals
                if should_check and bool(
                    ((primal <= eps_primal) & (dual <= eps_dual)).item()
                ):
                    status = "solved"
                    break
                if should_update_rho:
                    updated, rho_bar, rho_vec = _adaptive_rho_update(
                        rho_bar,
                        equality,
                        primal,
                        dual,
                        eps_primal,
                        eps_dual,
                        rho_tolerance,
                    )
                    if updated:
                        rho_updates += 1
                        K = build_kkt_matrix(P_s, A_s, sigma, rho_vec)
                        solver.refactorize(K)
        else:
            iteration = max_iter

        x_o, z_o, y_o = _unscale_state(x, z, y, scaling)
        if last_residuals is None:
            last_residuals = _residuals(
                P,
                q,
                A,
                x_o,
                z_o,
                y_o,
                float(settings["eps_abs"]),
                float(settings["eps_rel"]),
            )

        polish_info = _default_polish_info(settings)
        if bool(settings.get("polishing", True)) and status == "solved":
            x_o, z_o, y_o, polish_info = _polish_solution(
                P, q, A, l, u, x_o, z_o, y_o, settings
            )
            if polish_info["polishing_status"] == "rejected_no_improvement":
                raise TorchLinearSolveError(
                    "Requested polishing failed to produce an acceptable KKT point."
                )
            last_residuals = _residuals(
                P,
                q,
                A,
                x_o,
                z_o,
                y_o,
                float(settings["eps_abs"]),
                float(settings["eps_rel"]),
            )
        elif bool(settings.get("polishing", True)):
            polish_info["polishing_status"] = "skipped_unsolved"

        workspace.state = {
            "x": x_o.detach().clone(),
            "z": z_o.detach().clone(),
            "y": y_o.detach().clone(),
        }
        workspace.rho_bar = float(rho_bar.item())
        workspace.rho_setting = float(settings["rho"])
        primal, dual, eps_primal, eps_dual = last_residuals
        objective = 0.5 * torch.dot(x_o, P @ x_o) + torch.dot(q, x_o)
        info = {
            "status": status,
            "status_compatible": status == "solved",
            "admm_iterations": int(iteration),
            "primal_residual": float(primal.item()),
            "dual_residual": float(dual.item()),
            "eps_primal": float(eps_primal.item()),
            "eps_dual": float(eps_dual.item()),
            "objective": float(objective.item()),
            "backend": "torch",
            "linear_solver": DenseLUSolver.solver_name,
            "factorization_reused": factorization_reused,
            "factorizations_this_solve": solver.factorization_count - factors_before,
            "linear_solves_this_solve": solver.solve_count - solves_before,
            "factorization_count": solver.factorization_count,
            "linear_solve_count": solver.solve_count,
            "latest_linear_diagnostics": latest_linear_diagnostics,
            "maximum_linear_residual": maximum_linear_residual,
            "rho_updates": rho_updates,
            "rho_bar": float(rho_bar.item()),
            "rho_min": float(torch.min(rho_vec).item()),
            "rho_max": float(torch.max(rho_vec).item()),
            "scaling_applied": int(settings.get("scaling", 0)) > 0,
            "scaling_passes": int(settings.get("scaling", 0)),
            "device": str(q.device),
            "dtype": str(q.dtype),
            "estimated_kkt_dim": int(n + m),
            **polish_info,
        }
        if bool(settings.get("check_condition", False)):
            try:
                info["estimated_kkt_condition"] = float(torch.linalg.cond(K).item())
            except (NotImplementedError, RuntimeError):
                info["estimated_kkt_condition"] = None
        if bool(settings.get("return_state", False)):
            info["state"] = {
                key: value.detach().clone() for key, value in workspace.state.items()
            }
        workspace.last_info = dict(info)
        solution = x_o.reshape(n, 1)
        return (solution, info) if bool(settings.get("return_info", False)) else solution


def solve_torch_osqp(P, q, A, l, u, settings, workspace=None):
    return solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)


def solve_torch_osqp_dense(P, q, A, l, u, settings, workspace=None):
    """Compatibility alias for the archived prototype name."""
    return solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)


def _validate_qp(P, q, A, l, u, settings):
    tensors = {
        "P": _dense_tensor(P, "P"),
        "q": _dense_tensor(q, "q").reshape(-1),
        "A": _dense_tensor(A, "A"),
        "l": _dense_tensor(l, "l").reshape(-1),
        "u": _dense_tensor(u, "u").reshape(-1),
    }
    P, q, A, l, u = (tensors[key] for key in ("P", "q", "A", "l", "u"))
    if q.dtype not in {torch.float32, torch.float64}:
        raise ValueError("Torch OSQP supports float32 and float64 only.")
    for name, tensor in tensors.items():
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on {q.device}, got {tensor.device}.")
        if tensor.dtype != q.dtype:
            raise ValueError(f"{name} must use {q.dtype}, got {tensor.dtype}.")
    n = q.numel()
    m = l.numel()
    if n == 0:
        raise ValueError("Torch OSQP requires at least one variable.")
    if P.shape != (n, n):
        raise ValueError(f"P must have shape {(n, n)}, got {tuple(P.shape)}.")
    if A.ndim != 2 or A.shape != (m, n):
        raise ValueError(f"A must have shape {(m, n)}, got {tuple(A.shape)}.")
    if u.numel() != m:
        raise ValueError("l and u must have the same number of entries.")
    for name, tensor in (("P", P), ("q", q), ("A", A)):
        if not bool(torch.all(torch.isfinite(tensor)).item()):
            raise ValueError(f"{name} contains NaN or Inf.")
    if bool(torch.any(torch.isnan(l)).item()) or bool(torch.any(torch.isnan(u)).item()):
        raise ValueError("l and u must not contain NaN.")
    if bool(torch.any(l > u).item()):
        raise ValueError("Every lower bound must be less than or equal to its upper bound.")

    eps = torch.finfo(q.dtype).eps
    multiplier = float(settings.get("symmetry_tolerance_multiplier", 100.0))
    scale = max(1.0, float(torch.linalg.matrix_norm(P, ord=float("inf")).item()))
    asymmetry = float(torch.linalg.matrix_norm(P - P.T, ord=float("inf")).item())
    tolerance = multiplier * eps * scale
    if asymmetry > tolerance:
        raise ValueError(
            "P is materially asymmetric: "
            f"||P-P.T||_inf={asymmetry:.3e} exceeds {tolerance:.3e}."
        )
    P = 0.5 * (P + P.T)
    if bool(settings.get("check_convexity", False)):
        eigenvalues = torch.linalg.eigvalsh(P)
        spectral_scale = max(1.0, float(torch.max(torch.abs(eigenvalues)).item()))
        psd_tolerance = multiplier * eps * spectral_scale
        minimum = float(torch.min(eigenvalues).item())
        if minimum < -psd_tolerance:
            raise ValueError(
                f"P is not positive semidefinite: lambda_min={minimum:.3e}, "
                f"tolerance={psd_tolerance:.3e}."
            )
    return P, q, A, l, u


def _dense_tensor(value, name):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a Torch tensor.")
    tensor = value.detach()
    return tensor.to_dense() if tensor.layout != torch.strided else tensor


def _prepare_workspace(workspace, P, A, constraint_order_signature=None):
    signature = (tuple(P.shape), tuple(A.shape), str(P.device), str(P.dtype), "torch")
    p_pattern = P.ne(0)
    a_pattern = A.ne(0)
    compatible = (
        workspace.problem_signature == signature
        and workspace.constraint_order_signature == constraint_order_signature
        and workspace.p_pattern is not None
        and workspace.a_pattern is not None
        and torch.equal(workspace.p_pattern, p_pattern)
        and torch.equal(workspace.a_pattern, a_pattern)
    )
    if not compatible:
        workspace.reset_torch()
        workspace.problem_signature = signature
        workspace.constraint_order_signature = constraint_order_signature
        workspace.p_pattern = p_pattern.detach().clone()
        workspace.a_pattern = a_pattern.detach().clone()
        return
    scaling_matches = (
        workspace.scaling_source_p is not None
        and workspace.scaling_source_a is not None
        and torch.equal(workspace.scaling_source_p, P)
        and torch.equal(workspace.scaling_source_a, A)
    )
    if not scaling_matches:
        workspace.scaling = None
        workspace.scaling_source_p = None
        workspace.scaling_source_a = None
        workspace.scaling_passes = 0


def _scaling_for_problem(workspace, P, q, A, settings):
    passes = int(settings.get("scaling", 0))
    if passes <= 0:
        return {
            "D": torch.ones(P.shape[0], device=P.device, dtype=P.dtype),
            "E": torch.ones(A.shape[0], device=A.device, dtype=A.dtype),
            "cost": torch.ones((), device=P.device, dtype=P.dtype),
            "passes": 0,
        }
    if workspace.scaling is not None and workspace.scaling_passes == passes:
        return workspace.scaling
    tiny = torch.as_tensor(torch.finfo(P.dtype).tiny, device=P.device, dtype=P.dtype)
    D = torch.ones(P.shape[0], device=P.device, dtype=P.dtype)
    E = torch.ones(A.shape[0], device=A.device, dtype=A.dtype)
    P_work = P.clone()
    A_work = A.clone()
    q_work = q.clone()
    for _ in range(passes):
        p_norm = torch.maximum(
            torch.amax(torch.abs(P_work), dim=0),
            torch.amax(torch.abs(P_work), dim=1),
        )
        a_col = torch.amax(torch.abs(A_work), dim=0)
        a_row = torch.amax(torch.abs(A_work), dim=1)
        d_step = torch.rsqrt(torch.maximum(p_norm, a_col).clamp_min(tiny))
        e_step = torch.rsqrt(a_row.clamp_min(tiny))
        d_step = d_step.clamp(0.1, 10.0)
        e_step = e_step.clamp(0.1, 10.0)
        P_work = d_step[:, None] * P_work * d_step[None, :]
        A_work = e_step[:, None] * A_work * d_step[None, :]
        q_work = d_step * q_work
        D = D * d_step
        E = E * e_step
    one = torch.ones((), device=P.device, dtype=P.dtype)
    norm = torch.maximum(
        torch.amax(torch.abs(P_work)),
        torch.linalg.vector_norm(q_work, ord=float("inf")),
    )
    cost = one / torch.maximum(norm, one)
    scaling = {"D": D, "E": E, "cost": cost, "passes": passes}
    workspace.scaling = scaling
    workspace.scaling_source_p = P.detach().clone()
    workspace.scaling_source_a = A.detach().clone()
    workspace.scaling_passes = passes
    return scaling


def _scale_problem(P, q, A, l, u, scaling):
    D, E, cost = scaling["D"], scaling["E"], scaling["cost"]
    return (
        cost * (D[:, None] * P * D[None, :]),
        cost * D * q,
        E[:, None] * A * D[None, :],
        E * l,
        E * u,
    )


def _initial_scaled_state(workspace, n, m, device, dtype, scaling, warm_start):
    x_o = torch.zeros(n, device=device, dtype=dtype)
    z_o = torch.zeros(m, device=device, dtype=dtype)
    y_o = torch.zeros(m, device=device, dtype=dtype)
    if warm_start and isinstance(workspace.state, dict):
        x_o = _state_vector(workspace.state.get("x"), x_o)
        z_o = _state_vector(workspace.state.get("z"), z_o)
        y_o = _state_vector(workspace.state.get("y"), y_o)
    D, E, cost = scaling["D"], scaling["E"], scaling["cost"]
    return x_o / D, z_o * E, y_o * cost / E


def _state_vector(value, fallback):
    if not torch.is_tensor(value):
        return fallback
    value = value.detach().to(device=fallback.device, dtype=fallback.dtype).reshape(-1)
    if value.numel() != fallback.numel() or not bool(torch.all(torch.isfinite(value)).item()):
        return fallback
    return value.clone()


def _unscale_state(x, z, y, scaling):
    D, E, cost = scaling["D"], scaling["E"], scaling["cost"]
    return D * x, z / E, E * y / cost


def _equality_mask(l, u):
    finite = torch.isfinite(l) & torch.isfinite(u)
    atol = 100.0 * torch.finfo(l.dtype).eps
    return finite & torch.isclose(l, u, rtol=atol, atol=atol)


def _rho_vector(rho_bar, equality):
    rho_vec = torch.full(
        equality.shape,
        float(rho_bar.item()),
        device=rho_bar.device,
        dtype=rho_bar.dtype,
    )
    rho_vec[equality] *= 1000.0
    return rho_vec


def _adaptive_rho_update(rho_bar, equality, primal, dual, eps_primal, eps_dual, tolerance):
    tiny = torch.as_tensor(
        torch.finfo(rho_bar.dtype).tiny,
        device=rho_bar.device,
        dtype=rho_bar.dtype,
    )
    ratio = (primal / eps_primal.clamp_min(tiny)) / (
        dual / eps_dual.clamp_min(tiny)
    ).clamp_min(tiny)
    if not bool(((ratio > tolerance) | (ratio < 1.0 / tolerance)).item()):
        return False, rho_bar, _rho_vector(rho_bar, equality)
    rho_bar = (rho_bar * torch.sqrt(ratio).clamp(0.1, 10.0)).clamp(1e-6, 1e6)
    return True, rho_bar, _rho_vector(rho_bar, equality)


def _residuals(P, q, A, x, z, y, eps_abs, eps_rel):
    Ax, Px, ATy = A @ x, P @ x, A.T @ y
    primal = torch.linalg.vector_norm(Ax - z, ord=float("inf"))
    dual = torch.linalg.vector_norm(Px + q + ATy, ord=float("inf"))
    eps_primal = eps_abs + eps_rel * torch.maximum(
        torch.linalg.vector_norm(Ax, ord=float("inf")),
        torch.linalg.vector_norm(z, ord=float("inf")),
    )
    eps_dual = eps_abs + eps_rel * torch.maximum(
        torch.maximum(
            torch.linalg.vector_norm(Px, ord=float("inf")),
            torch.linalg.vector_norm(ATy, ord=float("inf")),
        ),
        torch.linalg.vector_norm(q, ord=float("inf")),
    )
    return primal, dual, eps_primal, eps_dual


def _default_polish_info(settings):
    enabled = bool(settings.get("polishing", True))
    return {
        "polishing": enabled,
        "polishing_success": False,
        "polishing_status": "not_run" if enabled else "disabled",
        "polishing_time_ms": 0.0,
        "polishing_active_constraints": 0,
        "polishing_refine_iter": int(settings.get("polish_refine_iter", 3)),
    }


def _polish_solution(P, q, A, l, u, x, z, y, settings):
    info = _default_polish_info(settings)
    started = time.perf_counter()
    equality = _equality_mask(l, u)
    activity_tolerance = 10.0 * (
        float(settings["eps_abs"])
        + float(settings["eps_rel"])
        * torch.maximum(torch.abs(z), torch.ones_like(z))
    )
    at_lower = torch.isfinite(l) & (torch.abs(z - l) <= activity_tolerance)
    at_upper = torch.isfinite(u) & (torch.abs(z - u) <= activity_tolerance)
    active = equality | ((y < 0) & at_lower) | ((y > 0) & at_upper)
    equality_indices = torch.nonzero(equality & active, as_tuple=False).reshape(-1)
    other_indices = torch.nonzero(active & ~equality, as_tuple=False).reshape(-1)
    indices = torch.cat((equality_indices, other_indices))
    info["polishing_active_constraints"] = int(indices.numel())
    if indices.numel() == 0:
        info["polishing_status"] = "skipped_no_active_constraints"
        info["polishing_time_ms"] = (time.perf_counter() - started) * 1000
        return x, z, y, info

    A_active = A[indices]
    rhs_active = torch.where(
        equality[indices],
        0.5 * (l[indices] + u[indices]),
        torch.where(y[indices] < 0, l[indices], u[indices]),
    )
    indices, A_active, rhs_active = _drop_duplicate_active_rows(
        indices, A_active, rhs_active
    )
    info["polishing_active_constraints"] = int(indices.numel())
    delta = float(settings.get("polish_delta", 1e-6))
    n, p = q.numel(), indices.numel()
    K_exact = torch.cat(
        (
            torch.cat(
                (P, A_active.T),
                dim=1,
            ),
            torch.cat(
                (
                    A_active,
                    torch.zeros((p, p), device=q.device, dtype=q.dtype),
                ),
                dim=1,
            ),
        ),
        dim=0,
    )
    regularizer = torch.cat(
        (
            torch.cat(
                (
                    delta * torch.eye(n, device=q.device, dtype=q.dtype),
                    torch.zeros((n, p), device=q.device, dtype=q.dtype),
                ),
                dim=1,
            ),
            torch.cat(
                (
                    torch.zeros((p, n), device=q.device, dtype=q.dtype),
                    -delta * torch.eye(p, device=q.device, dtype=q.dtype),
                ),
                dim=1,
            ),
        ),
        dim=0,
    )
    K_regularized = K_exact + regularizer
    rhs = torch.cat((-q, rhs_active))
    polish_solver = DenseLUSolver()
    try:
        try:
            polish_solver.factorize(K_exact)
            polished, _ = polish_solver.solve(
                rhs,
                calculate_residual=bool(
                    settings.get("check_linear_residual", False)
                ),
            )
            info["polishing_regularized"] = False
        except TorchLinearSolveError:
            polish_solver.factorize(K_regularized)
            polished, _ = polish_solver.solve(
                rhs,
                calculate_residual=bool(
                    settings.get("check_linear_residual", False)
                ),
            )
            for _ in range(int(settings.get("polish_refine_iter", 3))):
                correction, _ = polish_solver.solve(rhs - K_exact @ polished)
                polished = polished + correction
            info["polishing_regularized"] = True
    except (ValueError, TorchLinearSolveError) as exc:
        raise TorchLinearSolveError(f"Requested polishing failed: {exc}") from exc

    x_candidate = polished[:n]
    y_candidate = torch.zeros_like(y)
    y_candidate[indices] = polished[n:]
    z_candidate = torch.maximum(torch.minimum(A @ x_candidate, u), l)
    old = _residuals(
        P, q, A, x, z, y, float(settings["eps_abs"]), float(settings["eps_rel"])
    )
    new = _residuals(
        P,
        q,
        A,
        x_candidate,
        z_candidate,
        y_candidate,
        float(settings["eps_abs"]),
        float(settings["eps_rel"]),
    )
    tiny = torch.as_tensor(torch.finfo(q.dtype).tiny, device=q.device, dtype=q.dtype)
    old_score = torch.maximum(old[0] / old[2].clamp_min(tiny), old[1] / old[3].clamp_min(tiny))
    new_score = torch.maximum(new[0] / new[2].clamp_min(tiny), new[1] / new[3].clamp_min(tiny))
    satisfies = bool(((new[0] <= new[2]) & (new[1] <= new[3])).item())
    if not satisfies and not bool((new_score <= old_score).item()):
        info.update(
            {
                "polishing_status": "rejected_no_improvement",
                "polishing_old_score": float(old_score.item()),
                "polishing_new_score": float(new_score.item()),
                "polishing_time_ms": (time.perf_counter() - started) * 1000,
            }
        )
        return x, z, y, info
    info.update(
        {
            "polishing_success": True,
            "polishing_status": "accepted",
            "polishing_old_score": float(old_score.item()),
            "polishing_new_score": float(new_score.item()),
            "polishing_time_ms": (time.perf_counter() - started) * 1000,
        }
    )
    return x_candidate, z_candidate, y_candidate, info


def _drop_duplicate_active_rows(indices, rows, rhs):
    """Keep equality-first representatives of identical active constraints."""

    if indices.numel() <= 1:
        return indices, rows, rhs
    augmented = torch.cat((rows, rhs[:, None]), dim=1)
    _, inverse = torch.unique(augmented, dim=0, return_inverse=True)
    positions = torch.arange(rows.shape[0], device=rows.device, dtype=torch.long)
    first = torch.full(
        (int(torch.max(inverse).item()) + 1,),
        rows.shape[0],
        device=rows.device,
        dtype=torch.long,
    )
    first.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
    keep_tensor = torch.sort(first).values
    return indices[keep_tensor], rows[keep_tensor], rhs[keep_tensor]
