"""Dense Torch reference implementation of OSQP's ADMM equations."""

from __future__ import annotations

import math
import time

import torch

from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchLinearSolve import TorchLinearSolveError
from pygranso.private.torchQDLDLSolve import QDLDLSolver
from pygranso.private.torchSparseSolve import SparseDirectSolver

_clock = time.perf_counter

_STATUS_VALUES = {
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

_SOLUTION_STATUSES = {
    "solved",
    "solved_inaccurate",
    "max_iter_reached",
    "time_limit_reached",
}
_COMPATIBLE_STATUSES = {"solved", "solved_inaccurate"}
# osqp_api_constants.h casts the integer literal to OSQPFloat.
_OSQP_NAN = float(0x7FC00000)

_RHO_DISABLED = 0
_RHO_ITERATIONS = 1
_RHO_TIME = 2
_RHO_KKT_ERROR = 3


def build_kkt_matrix(P, A, sigma, rho_vec):
    n = P.shape[0]
    identity = torch.eye(n, device=P.device, dtype=P.dtype)
    top = torch.cat((P + float(sigma) * identity, A.T), dim=1)
    bottom = torch.cat((A, -torch.diag(rho_vec.reciprocal())), dim=1)
    return torch.cat((top, bottom), dim=0)


def build_kkt_rhs(x, z, y, q, sigma, rho_inv_vec):
    rho_inv_y = rho_inv_vec * y
    return torch.cat((float(sigma) * x - q, -rho_inv_y + z))


def recover_z_tilde(rhs_lower, nu, rho_inv_vec):
    return rhs_lower + rho_inv_vec * nu


def admm_vector_update(
    x_tilde, x, z_tilde, z, y, rho_vec, l, u, alpha, *,
    rho_inv_vec, return_delta_y=False,
):
    x_next = float(alpha) * x_tilde + (1.0 - float(alpha)) * x
    z_relaxed = float(alpha) * z_tilde + (1.0 - float(alpha)) * z
    z_argument = rho_inv_vec * y + z_relaxed
    z_next = torch.maximum(torch.minimum(z_argument, u), l)
    delta_y = rho_vec * (z_relaxed - z_next)
    y_next = y + delta_y
    if return_delta_y:
        return x_next, z_next, y_next, delta_y
    return x_next, z_next, y_next


def _setup_rho_interval(mode, requested_interval, check_termination):
    """Apply the pinned setup-time interval rules for an adaptive-rho mode."""

    if mode == _RHO_ITERATIONS and requested_interval == 0:
        return 4 * check_termination if check_termination else 100
    if mode == _RHO_KKT_ERROR and requested_interval == 0:
        return 1
    return requested_interval


def _time_rho_interval(
    *, iteration, check_termination, elapsed, setup_time, fraction
):
    """Resolve the pinned profiling-time interval after its strict threshold."""

    if elapsed <= fraction * setup_time:
        return 0
    multiple = check_termination if check_termination else 25
    rounded = int(
        iteration
        + 0.5 * multiple
        - ((iteration + 0.5 * multiple) % multiple)
    )
    # This intentionally translates the executable c_min/c_max expression in
    # pinned osqp_api.c, even though its adjacent comment describes a minimum.
    return min(1, max(rounded, check_termination))


def _relative_kkt_error(primal_residual, dual_residual, duality_gap):
    """Translate auxil.c::update_info, including its signed gap semantics."""

    return torch.maximum(
        torch.maximum(dual_residual, primal_residual), duality_gap
    )


def _kkt_rho_eligible(relative_kkt_error, fraction, last_relative_kkt_error):
    return bool(
        (
            relative_kkt_error
            <= float(fraction) * float(last_relative_kkt_error)
        ).item()
    )


def solve_torch_osqp_direct(P, q, A, l, u, settings, workspace=None):
    """Solve ``min 0.5*x'Px + q'x`` subject to ``l <= Ax <= u``."""

    workspace = workspace or TorchOSQPWorkspace()
    snapshot = _snapshot_workspace(workspace)
    try:
        result = _solve_torch_osqp_direct_impl(P, q, A, l, u, settings, workspace)
    except BaseException:
        _restore_workspace(workspace, snapshot)
        raise

    return result


def _solve_torch_osqp_direct_impl(P, q, A, l, u, settings, workspace):
    with torch.no_grad():
        solve_started = _clock()
        P, q, A, l, u = _validate_qp(P, q, A, l, u, settings)
        if settings.get("rho_update_safeguard", False):
            raise ValueError("The pinned C rho update requires rho_update_safeguard=False.")
        use_sparse = settings.get("use_sparse", False)
        if not isinstance(use_sparse, bool):
            raise ValueError("use_sparse must be a bool.")
        solver_type = SparseDirectSolver if use_sparse else QDLDLSolver
        if type(workspace.linear_solver) is not solver_type:
            # A package change is a fresh setup, not a reuse of foreign factors.
            workspace.reset_torch()
            workspace.linear_solver = solver_type()
        rho_mode = int(settings.get("adaptive_rho", True))
        requested_rho_interval = int(settings.get("rho_update_interval", 50))
        rho_fraction = float(settings.get("adaptive_rho_fraction", 0.4))
        rho_tolerance = float(settings.get("rho_update_tolerance", 5.0))
        rho_setup_signature = (
            rho_mode,
            requested_rho_interval,
            rho_fraction,
            rho_tolerance,
        )
        if (
            workspace.rho_setup_signature is not None
            and workspace.rho_setup_signature != rho_setup_signature
        ):
            # The pinned update-settings API ignores these setup-only fields.
            # A direct Torch request for different values is a fresh setup.
            workspace.reset_torch()
        _prepare_workspace(
            workspace,
            P,
            A,
            settings.get("_constraint_order_signature"),
            l,
            u,
            int(settings.get("scaling", 0)),
            track_constraint_types=False,
        )
        # Repeated solves supply implicit value-changing data updates.
        problem = (P, q, A, l, u)
        rho_updates_base = workspace.rho_updates_committed
        if workspace.status_problem is None or any(
            not torch.equal(previous, current)
            for previous, current in zip(workspace.status_problem, problem)
        ):
            workspace.status = "unsolved"
            rho_updates_base = 0
            workspace.status_problem = tuple(value.detach().clone() for value in problem)
        fresh_rho_setup = workspace.rho_setup_signature is None
        if fresh_rho_setup:
            workspace.adaptive_rho_interval = _setup_rho_interval(
                rho_mode, requested_rho_interval, int(settings["check_termination"])
            )
            workspace.last_rel_kkt = _osqp_infinity(q)
        workspace.rho_setup_signature = rho_setup_signature
        scaling = _scaling_for_problem(workspace, P, q, A, settings)
        P_s, q_s, A_s, l_s, u_s = _scale_problem(P, q, A, l, u, scaling)
        if scaling["bounds_set"] or workspace.constraint_types is None:
            # set_rho_vec runs in osqp_setup and in osqp_update_data_vec(l, u).
            constraint_types = _constraint_types(l_s, u_s)
        else:
            # osqp_update_data_mat rescales the bounds without revisiting types.
            constraint_types = workspace.constraint_types
        # A type change clears the factors but can leave the validated top rows unchanged.
        previous_k = workspace.linear_solver.matrix
        _reconcile_workspace_constraint_types(workspace, constraint_types)

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
            A_s,
        )
        # A cold start clears the iterates but keeps the active rho.
        requested_rho = float(settings["rho"])
        initial_rho = (
            workspace.rho_bar
            if workspace.rho_bar is not None and workspace.rho_setting == requested_rho
            else requested_rho
        )
        rho_bar = torch.as_tensor(
            initial_rho, device=q.device, dtype=q.dtype
        ).clamp(1e-6, 1e6)
        equality = constraint_types == 1
        loose = constraint_types == -1
        rho_vec = _rho_vector(rho_bar, equality, loose)
        sigma = float(settings["sigma"])
        alpha = float(settings["alpha"])
        max_iter = int(settings["max_iter"])
        time_limit = float(settings.get("time_limit", 1e10))
        check_termination = int(settings["check_termination"])
        scaled_termination = bool(settings.get("scaled_termination", False))
        rho_interval = int(workspace.adaptive_rho_interval or 0)
        check_linear_residual = bool(settings.get("check_linear_residual", False))

        solver = workspace.linear_solver
        factors_before = solver.factorization_count
        solves_before = solver.solve_count
        K = build_kkt_matrix(P_s, A_s, sigma, rho_vec)
        # Use the represented reciprocals already stored in the KKT diagonal.
        rho_inv_vec = -K.diagonal()[n:]
        matrix_data_changed = (
            previous_k is None
            or previous_k.shape != K.shape
            or previous_k.device != K.device
            or previous_k.dtype != K.dtype
            or not torch.equal(previous_k[:n], K[:n])
        )
        if use_sparse:
            factorization_reused = not solver.factorize_if_needed(K)
            # Pinned rho-only updates check factor failure, not positive-D count.
            if matrix_data_changed:
                # A positive variable block and negative rho block establish the
                # expected KKT inertia without losing small signs in a full solve.
                eigenvalues = torch.linalg.eigvalsh(K[:n, :n])
                if not bool(torch.all(torch.isfinite(eigenvalues)).item()):
                    raise TorchLinearSolveError("KKT inertia check returned non-finite values.")
                if bool(torch.any(eigenvalues <= 0).item()):
                    # P alone is not the QDLDL setup/update criterion: constraints
                    # can change the inertia even when P + sigma I is indefinite.
                    eigenvalues = torch.linalg.eigvalsh(K)
                    if not bool(torch.all(torch.isfinite(eigenvalues)).item()):
                        raise TorchLinearSolveError(
                            "KKT inertia check returned non-finite values."
                        )
                if int(torch.count_nonzero(eigenvalues > 0).item()) != n or bool(
                    torch.any(eigenvalues == 0).item()
                ):
                    raise TorchLinearSolveError(
                        "KKT inertia does not have the required positive/negative "
                        "sign counts; the problem may be non-convex."
                    )
        else:
            factorization_reused = solver.is_factorized_for(K)
            if not factorization_reused:
                # init_linsys_solver_qdldl / update_linsys_solver_*_qdldl:
                # form_KKT, AMD-permute, QDLDL_etree, QDLDL_factor.
                solver.factorize(
                    K,
                    P=P_s,
                    A=A_s,
                    sigma=sigma,
                    rho_inv_vec=rho_inv_vec,
                    p_pattern=workspace.p_pattern,
                    a_pattern=workspace.a_pattern,
                )
            # LDL_factor (setup) rejects fewer than n positive pivots and
            # update_linsys_solver_matrices_qdldl any count other than n;
            # update_linsys_solver_rho_vec_qdldl only checks factor failure.
            if matrix_data_changed and (
                solver.positive_d_count < n
                if previous_k is None
                else solver.positive_d_count != n
            ):
                raise TorchLinearSolveError(
                    "KKT inertia does not have the required positive/negative "
                    "sign counts; the problem may be non-convex."
                )
        solve_clock_started = _clock()
        if fresh_rho_setup or workspace.adaptive_rho_setup_time is None:
            workspace.adaptive_rho_setup_time = solve_clock_started - solve_started
        rho_setup_time = workspace.adaptive_rho_setup_time
        maximum_linear_residual = None
        latest_linear_diagnostics = None
        status = workspace.status
        rho_updates = 0
        last_residuals = None
        should_check = False
        primal_certificate = None
        dual_certificate = None
        # True once check_termination writes a status during this osqp_solve.
        status_written = False
        delta_x_o = torch.zeros_like(x)
        delta_y_o = torch.zeros_like(y)
        delta_x_t = torch.zeros_like(x)
        delta_y_t = torch.zeros_like(y)
        primal_certificate_context = None
        dual_certificate_context = None

        for iteration in range(1, max_iter + 1):
            x_previous, z_previous, y_previous = x, z, y
            rhs = build_kkt_rhs(x, z, y, q_s, sigma, rho_inv_vec)
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
            z_tilde = recover_z_tilde(rhs[n:], nu, rho_inv_vec)
            x, z, y, delta_y_s = admm_vector_update(
                x_tilde,
                x,
                z_tilde,
                z,
                y,
                rho_vec,
                l_s,
                u_s,
                alpha,
                rho_inv_vec=rho_inv_vec,
                return_delta_y=True,
            )

            elapsed = _clock() - solve_started
            time_limit_reached = time_limit != 0.0 and elapsed >= time_limit
            # A profiling timeout breaks before C updates the periodic-check flag.
            if not time_limit_reached:
                should_check = (
                    check_termination > 0 and iteration % check_termination == 0
                )
            can_adapt_rho = False
            if not time_limit_reached and rho_mode != _RHO_DISABLED:
                if rho_mode == _RHO_TIME and rho_interval == 0:
                    rho_interval = _time_rho_interval(
                        iteration=iteration,
                        check_termination=check_termination,
                        elapsed=_clock() - solve_clock_started,
                        setup_time=rho_setup_time,
                        fraction=rho_fraction,
                    )
                    workspace.adaptive_rho_interval = rho_interval
                can_adapt_rho = bool(
                    rho_interval and iteration % rho_interval == 0
                )
            if (
                should_check
                or can_adapt_rho
                or iteration == 1
                or iteration == max_iter
                or time_limit_reached
            ):
                # auxil.c::update_y retains the increment before adding it to y.
                # Subtracting the rounded dual iterates can lose that increment.
                delta_x_s = x - x_previous
                delta_x_o = scaling["D"] * delta_x_s
                delta_y_o = scaling["E"] * delta_y_s / scaling["cost"]
                if scaled_termination:
                    P_t, q_t, A_t, l_t, u_t = P_s, q_s, A_s, l_s, u_s
                    delta_x_t, delta_y_t = delta_x_s, delta_y_s
                else:
                    P_t, q_t, A_t, l_t, u_t = P, q, A, l, u
                    delta_x_t, delta_y_t = delta_x_o, delta_y_o
                primal_certificate_context = (
                    A_s,
                    l_s,
                    u_s,
                    delta_y_s,
                    None if scaled_termination else scaling["E"],
                    None if scaled_termination else scaling["D"].reciprocal(),
                    None if scaled_termination else scaling["E"],
                )
                dual_certificate_context = (
                    P_s,
                    q_s,
                    A_s,
                    l_s,
                    u_s,
                    delta_x_s,
                    None if scaled_termination else scaling["D"],
                    None if scaled_termination else scaling["D"].reciprocal(),
                    None if scaled_termination else scaling["E"].reciprocal(),
                    None if scaled_termination else scaling["cost"],
                    None if scaled_termination else scaling["D"],
                )
                if time_limit_reached:
                    status = "time_limit_reached"
                    break
                last_residuals = _residuals(
                    P_s,
                    q_s,
                    A_s,
                    x,
                    z,
                    y,
                    float(settings["eps_abs"]),
                    float(settings["eps_rel"]),
                    scaling=None if scaled_termination else scaling,
                )
                primal, dual, eps_primal, eps_dual = last_residuals
                objective_values = _compute_obj_val_dual_gap(
                    P_s, q_s, l_s, u_s, x, y, scaling["cost"],
                    float(settings["eps_abs"]), float(settings["eps_rel"]),
                )
                if should_check:
                    termination = _termination_status(
                        P_t,
                        q_t,
                        A_t,
                        l_t,
                        u_t,
                        delta_x_t,
                        delta_y_t,
                        last_residuals,
                        settings,
                        approximate=False,
                        objective_values=objective_values,
                        dual_bounds=(l_s, u_s),
                        primal_certificate_context=primal_certificate_context,
                        dual_certificate_context=dual_certificate_context,
                    )
                    if termination is not None:
                        status, primal_certificate, dual_certificate = termination
                        status_written = True
                        break
                if can_adapt_rho and rho_mode == _RHO_KKT_ERROR:
                    rel_kkt_error = _relative_kkt_error(
                        primal, dual, objective_values["duality_gap"]
                    )
                    can_adapt_rho = _kkt_rho_eligible(
                        rel_kkt_error, rho_fraction, workspace.last_rel_kkt
                    )
                if can_adapt_rho:
                    scaled_residuals = _residuals(
                        P_s, q_s, A_s, x, z, y, 0.0, 0.0
                    )
                    scaled_primal, scaled_dual = scaled_residuals[:2]
                    primal_normalizer, dual_normalizer = _residual_normalizers(
                        P_s, q_s, A_s, x, z, y
                    )
                    updated, rho_candidate, rho_vec_candidate = _adaptive_rho_update(
                        rho_bar,
                        equality,
                        scaled_primal,
                        scaled_dual,
                        eps_primal,
                        eps_dual,
                        rho_tolerance,
                        loose=loose,
                        primal_normalizer=primal_normalizer,
                        dual_normalizer=dual_normalizer,
                    )
                    if updated:
                        candidate_k = build_kkt_matrix(
                            P_s, A_s, sigma, rho_vec_candidate
                        )
                        if use_sparse:
                            solver.refactorize(candidate_k)
                        else:
                            # update_linsys_solver_rho_vec_qdldl: new rho
                            # values, then QDLDL_factor on the same pattern.
                            solver.factorize(
                                candidate_k,
                                P=P_s,
                                A=A_s,
                                sigma=sigma,
                                rho_inv_vec=-candidate_k.diagonal()[n:],
                                p_pattern=workspace.p_pattern,
                                a_pattern=workspace.a_pattern,
                            )
                        K = candidate_k
                        rho_bar = rho_candidate
                        rho_vec = rho_vec_candidate
                        rho_inv_vec = -K.diagonal()[n:]
                        rho_updates += 1
                        if rho_mode != _RHO_KKT_ERROR:
                            rel_kkt_error = _relative_kkt_error(
                                primal, dual, objective_values["duality_gap"]
                            )
                        workspace.last_rel_kkt = float(rel_kkt_error.item())
        else:
            iteration = max_iter

        x_o, z_o, y_o = _unscale_state(x, z, y, scaling)
        if scaled_termination:
            P_t, q_t, A_t, l_t, u_t = P_s, q_s, A_s, l_s, u_s
        else:
            P_t, q_t, A_t, l_t, u_t = P, q, A, l, u
        timed_out = time_limit_reached
        retain_info = timed_out and should_check
        reported_iteration = iteration - 1 if timed_out else iteration
        if should_check:
            # The checked footer retains residuals; timeouts retain prior products.
            x_info = x_previous if timed_out else x
            y_info = y_previous if timed_out else y
            Ax, Px, ATy = (
                _matvec(A_s, x_info), _matvec(P_s, x_info), _rmatvec(A_s, y_info)
            )
            rel_kkt_error = _relative_kkt_error(
                *last_residuals[:2], objective_values["duality_gap"]
            )
            Dinv = Einv = None
            cost_inverse = 1.0
            if not scaled_termination:
                Dinv, Einv = scaling["D"].reciprocal(), scaling["E"].reciprocal()
                cost_inverse = scaling["cost"].reciprocal()
            normalizers = _termination_normalizers(
                Ax, Px, ATy, q_s, z, Dinv, Einv, cost_inverse
            )
            eps_abs = float(settings["eps_abs"])
            eps_rel = float(settings["eps_rel"])
            last_residuals = (
                *last_residuals[:2],
                *(eps_abs + eps_rel * norm for norm in normalizers),
            )
            objective_values = _compute_obj_val_dual_gap(
                P_s, q_s, l_s, u_s, x, y, scaling["cost"],
                float(settings["eps_abs"]), float(settings["eps_rel"]), Px=Px,
            )
        else:
            last_residuals, normalizers = _residuals(
                P_s,
                q_s,
                A_s,
                x,
                z,
                y,
                float(settings["eps_abs"]),
                float(settings["eps_rel"]),
                scaling=None if scaled_termination else scaling,
                return_normalizers=True,
            )
            objective_values = _compute_obj_val_dual_gap(
                P_s, q_s, l_s, u_s, x, y, scaling["cost"],
                float(settings["eps_abs"]), float(settings["eps_rel"]),
            )
        if not should_check or status in {"unsolved", "time_limit_reached"}:
            termination = None
            if not should_check:
                termination = _termination_status(
                    P_t,
                    q_t,
                    A_t,
                    l_t,
                    u_t,
                    delta_x_t,
                    delta_y_t,
                    last_residuals,
                    settings,
                    approximate=False,
                    objective_values=objective_values,
                    dual_bounds=(l_s, u_s),
                    primal_certificate_context=primal_certificate_context,
                    dual_certificate_context=dual_certificate_context,
                )
            if termination is None and status in {"unsolved", "time_limit_reached"}:
                # auxil.c scales each OSQPFloat epsilon before forming tolerances.
                eps_abs = last_residuals[0].new_tensor(settings["eps_abs"]) * 10
                eps_rel = last_residuals[0].new_tensor(settings["eps_rel"]) * 10
                approximate_residuals = (
                    *last_residuals[:2],
                    *(eps_abs + eps_rel * norm for norm in normalizers),
                )
                approximate_objective = {
                    **objective_values,
                    "eps_duality_gap": (
                        eps_abs + eps_rel * objective_values["gap_normalizer"]
                    ),
                    "eps_scaled_duality_gap": (
                        eps_abs + eps_rel * objective_values["scaled_gap_normalizer"]
                    ),
                }
                termination = _termination_status(
                    P_t,
                    q_t,
                    A_t,
                    l_t,
                    u_t,
                    delta_x_t,
                    delta_y_t,
                    approximate_residuals,
                    settings,
                    approximate=True,
                    objective_values=approximate_objective,
                    dual_bounds=(l_s, u_s),
                    primal_certificate_context=primal_certificate_context,
                    dual_certificate_context=dual_certificate_context,
                )
            if termination is not None:
                status, primal_certificate, dual_certificate = termination
                status_written = True
            elif status == "unsolved":
                status = "max_iter_reached"

        if retain_info:
            # The final estimate reads the same cached scaled residuals/products.
            rho_residuals = _inf_norm(Ax - z_previous), _inf_norm(q_s + Px + ATy)
            rho_normalizers = _termination_normalizers(Ax, Px, ATy, q_s, z, None, None, 1.0)
        else:
            rho_residuals = _residuals(P_s, q_s, A_s, x, z, y, 0.0, 0.0)
            rho_normalizers = _residual_normalizers(P_s, q_s, A_s, x, z, y)
        rho_estimate = _compute_rho_estimate(
            rho_bar,
            rho_residuals[0],
            rho_residuals[1],
            rho_normalizers[0],
            rho_normalizers[1],
        )

        polish_info = _default_polish_info(settings)
        # osqp_setup initializes status_polish. reset_info preserves it on updates.
        x_work, z_work, y_work = x, z, y
        if bool(settings.get("polishing", False)) and status == "solved":
            x_polished_s, z_polished_s, y_polished_s, polish_info = _polish_solution(
                P_s,
                q_s,
                A_s,
                l_s,
                u_s,
                x,
                z,
                y,
                settings,
                scaling=scaling,
                p_pattern=workspace.p_pattern,
                a_pattern=workspace.a_pattern,
            )
            x_work, z_work, y_work = x_polished_s, z_polished_s, y_polished_s
            x_o, z_o, y_o = _unscale_state(
                x_polished_s, z_polished_s, y_polished_s, scaling
            )
            last_residuals = _residuals(
                P_s,
                q_s,
                A_s,
                x_polished_s,
                z_polished_s,
                y_polished_s,
                float(settings["eps_abs"]),
                float(settings["eps_rel"]),
                scaling=None if scaled_termination else scaling,
            )
            workspace.status_polish = {
                "accepted": 1,
                "rejected_no_improvement": -1,
                "skipped_no_active_constraints": 2,
            }[polish_info["polishing_status"]]
            if polish_info["polishing_success"]:
                objective_values = _compute_obj_val_dual_gap(
                    P_s, q_s, l_s, u_s, x_polished_s, y_polished_s,
                    scaling["cost"],
                    float(settings["eps_abs"]), float(settings["eps_rel"]),
                )
        elif bool(settings.get("polishing", False)):
            polish_info["polishing_status"] = "skipped_unsolved"

        has_iterate = status in _SOLUTION_STATUSES
        if has_iterate:
            workspace.state = {
                "x": x_o.detach().clone(),
                "z": z_o.detach().clone(),
                "y": y_o.detach().clone(),
            }
            # osqp_solve leaves work->x, work->z, work->y in scaled coordinates
            # for the next solve; polish() overwrites them only when it succeeds.
            workspace.work_iterates = {
                "x": x_work.detach().clone(),
                "z": z_work.detach().clone(),
                "y": y_work.detach().clone(),
                "source": workspace.state,
            }
        else:
            # auxil.c::store_solution calls osqp_cold_start when no solution exists.
            workspace.state = None
            workspace.work_iterates = None
        # osqp_update_rho writes settings->rho, which survives every status.
        workspace.rho_bar = float(rho_bar.item())
        workspace.rho_setting = float(settings["rho"])
        primal, dual, eps_primal, eps_dual = last_residuals
        if not retain_info:
            rel_kkt_error = _relative_kkt_error(
                primal, dual, objective_values["duality_gap"]
            )
        # check_termination overwrites info->obj_val only when it detects the
        # condition in this solve.  A status carried over from an earlier solve
        # leaves the value written by the last update_info.
        if status_written and status.startswith("primal_infeasible"):
            objective = _osqp_infinity(q)
        elif status_written and status.startswith("dual_infeasible"):
            objective = -_osqp_infinity(q)
        elif status_written and status == "non_convex":
            objective = _OSQP_NAN
        else:
            objective = float(objective_values["objective"].item())
        if not status_written and status.startswith("primal_infeasible"):
            # auxil.c::store_solution normalizes work->delta_y whatever wrote the
            # status.  is_primal_infeasible projects delta_y in place whenever
            # the final check found the primal residual above its tolerance.
            stale_delta_y = delta_y_s
            final_check_ran = (should_check and not timed_out) or not should_check
            if final_check_ran and m > 0 and not bool((primal < eps_primal).item()):
                stale_delta_y = _project_polar_reccone(stale_delta_y, l_s, u_s)
            primal_certificate = stale_delta_y * _inf_norm(stale_delta_y).reciprocal()
        if not status_written and status.startswith("dual_infeasible"):
            stale_delta_x = x - x_previous
            dual_certificate = stale_delta_x * _inf_norm(stale_delta_x).reciprocal()
        primal_certificate_result = (
            None if primal_certificate is None else primal_certificate.detach().clone()
        )
        dual_certificate_result = (
            None if dual_certificate is None else dual_certificate.detach().clone()
        )
        info = {
            "status": status,
            "status_val": _STATUS_VALUES[status],
            "status_compatible": status in _COMPATIBLE_STATUSES,
            "admm_iterations": int(reported_iteration),
            "primal_residual": float(primal.item()),
            "dual_residual": float(dual.item()),
            "eps_primal": float(eps_primal.item()),
            "eps_dual": float(eps_dual.item()),
            "objective": objective,
            "dual_objective": float(objective_values["dual_objective"].item()),
            "duality_gap": float(objective_values["duality_gap"].item()),
            "rel_kkt_error": float(rel_kkt_error.item()),
            "eps_duality_gap": float(objective_values["eps_duality_gap"].item()),
            "scaled_duality_gap": float(
                objective_values["scaled_duality_gap"].item()
            ),
            "eps_scaled_duality_gap": float(
                objective_values["eps_scaled_duality_gap"].item()
            ),
            "check_dualgap": settings.get("check_dualgap", q.dtype == torch.float64),
            "scaled_termination": scaled_termination,
            "primal_infeasibility_certificate": primal_certificate_result,
            "dual_infeasibility_certificate": dual_certificate_result,
            "prim_inf_cert": primal_certificate_result,
            "dual_inf_cert": dual_certificate_result,
            "backend": "torch",
            "linear_solver": solver.solver_name,
            "factorization_reused": factorization_reused and not use_sparse,
            "factorizations_this_solve": solver.factorization_count - factors_before,
            "linear_solves_this_solve": solver.solve_count - solves_before,
            "factorization_count": solver.factorization_count,
            "linear_solve_count": solver.solve_count,
            "latest_linear_diagnostics": latest_linear_diagnostics,
            "maximum_linear_residual": maximum_linear_residual,
            "rho_updates": rho_updates_base + rho_updates,
            "rho_bar": float(rho_bar.item()),
            "rho_estimate": float(rho_estimate.item()),
            "rho_min": (
                float(rho_bar.item())
                if rho_vec.numel() == 0
                else float(torch.min(rho_vec).item())
            ),
            "rho_max": (
                float(rho_bar.item())
                if rho_vec.numel() == 0
                else float(torch.max(rho_vec).item())
            ),
            "scaling_applied": int(settings.get("scaling", 0)) > 0,
            "scaling_passes": int(settings.get("scaling", 0)),
            "device": str(q.device),
            "dtype": str(q.dtype),
            "estimated_kkt_dim": int(n + m),
            "status_polish": workspace.status_polish,
            **polish_info,
        }
        if bool(settings.get("check_condition", False)):
            try:
                info["estimated_kkt_condition"] = float(torch.linalg.cond(K).item())
            except (NotImplementedError, RuntimeError):
                info["estimated_kkt_condition"] = None
        if bool(settings.get("return_state", False)) and has_iterate:
            info["state"] = {
                key: value.detach().clone() for key, value in workspace.state.items()
            }
        workspace.status = status
        workspace.rho_updates_committed = info["rho_updates"]
        workspace.last_info = dict(info)
        solution = (
            x_o.reshape(n, 1)
            if has_iterate
            else torch.full((n, 1), _OSQP_NAN, device=q.device, dtype=q.dtype)
        )
        return (
            (solution, info) if bool(settings.get("return_info", False)) else solution
        )


def solve_torch_osqp(P, q, A, l, u, settings, workspace=None):
    return solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)


def solve_torch_osqp_dense(P, q, A, l, u, settings, workspace=None):
    """Compatibility alias for the archived prototype name."""
    return solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)


def _snapshot_workspace(workspace):
    solver = workspace.linear_solver
    fields = workspace.__dict__.copy()
    fields["builtin_stats"] = dict(workspace.builtin_stats)
    return fields, solver, solver.__dict__.copy()


def _restore_workspace(workspace, snapshot):
    fields, solver, solver_fields = snapshot
    solver.__dict__.clear()
    solver.__dict__.update(solver_fields)
    workspace.__dict__.clear()
    workspace.__dict__.update(fields)


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
        raise ValueError(
            "Every lower bound must be less than or equal to its upper bound."
        )
    eps = torch.finfo(q.dtype).eps
    multiplier = float(settings.get("symmetry_tolerance_multiplier", 100.0))
    scale = max(1.0, float(torch.linalg.matrix_norm(P, ord=float("inf")).item()))
    tolerance = multiplier * eps * scale
    lower_triangle = torch.tril(P, diagonal=-1)
    if bool(torch.any(lower_triangle != 0).item()):
        asymmetry = float(torch.linalg.matrix_norm(P - P.T, ord=float("inf")).item())
        if asymmetry > tolerance:
            raise ValueError(
                "P is materially asymmetric: "
                f"||P-P.T||_inf={asymmetry:.3e} exceeds {tolerance:.3e}."
            )
        P = 0.5 * (P + P.T)
    else:
        # Pinned OSQP accepts upper-triangular storage and interprets it as
        # the symmetric matrix used by products and the KKT system.
        rows = torch.arange(n, device=P.device)
        P = torch.where(rows[:, None] > rows, P.T, P)
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


def _prepare_workspace(
    workspace,
    P,
    A,
    constraint_order_signature=None,
    l=None,
    u=None,
    scaling_setting=None,
    *,
    track_constraint_types=True,
):
    signature = (tuple(P.shape), tuple(A.shape), str(P.device), str(P.dtype), "torch")
    p_pattern = P.ne(0)
    a_pattern = A.ne(0)
    constraint_types = (
        None
        if not track_constraint_types or l is None or u is None
        else _constraint_types(l, u)
    )
    # Pinned l/u updates are vector updates. Bound values can change without
    # reordering the unchanged A rows or invalidating their KKT factorization;
    # equality/loose transitions are handled separately by constraint_types.
    constraint_rows = A
    rows_reordered = _rows_reordered(workspace.constraint_rows, constraint_rows)
    types_compatible = (
        constraint_types is None
        or workspace.constraint_types is None
        or torch.equal(workspace.constraint_types, constraint_types)
    )
    scaling_compatible = (
        scaling_setting is None
        or workspace.scaling_setting is None
        or workspace.scaling_setting == scaling_setting
    )
    compatible = (
        workspace.problem_signature == signature
        and workspace.constraint_order_signature == constraint_order_signature
        and workspace.p_pattern is not None
        and workspace.a_pattern is not None
        and torch.equal(workspace.p_pattern, p_pattern)
        and torch.equal(workspace.a_pattern, a_pattern)
        and not rows_reordered
        and scaling_compatible
    )
    if not compatible:
        workspace.reset_torch()
        workspace.problem_signature = signature
        workspace.constraint_order_signature = constraint_order_signature
        workspace.p_pattern = p_pattern.detach().clone()
        workspace.a_pattern = a_pattern.detach().clone()
        workspace.constraint_rows = constraint_rows.detach().clone()
        workspace.constraint_types = (
            None if constraint_types is None else constraint_types.detach().clone()
        )
        workspace.scaling_setting = scaling_setting
        return
    workspace.constraint_rows = constraint_rows.detach().clone()
    if track_constraint_types:
        if not types_compatible:
            workspace.linear_solver.clear()
        workspace.constraint_types = (
            None if constraint_types is None else constraint_types.detach().clone()
        )
    workspace.scaling_setting = scaling_setting
    has_scaling_source = (
        workspace.scaling_source_p is not None
        and workspace.scaling_source_a is not None
    )
    scaling_matches = (
        has_scaling_source
        and torch.equal(workspace.scaling_source_p, P)
        and torch.equal(workspace.scaling_source_a, A)
    )
    if has_scaling_source and not scaling_matches:
        workspace.linear_solver.clear()
    if not scaling_matches:
        if (
            has_scaling_source
            and workspace.scaling is not None
            and workspace.scaling_passes > 0
            and workspace.scaling_passes == scaling_setting
        ):
            # osqp_update_data_mat: the stored scaling is needed to unscale the
            # work data before the new values are inserted and rescaled.
            workspace.scaling = {
                **workspace.scaling,
                "matrix_update": (
                    not torch.equal(workspace.scaling_source_p, P),
                    not torch.equal(workspace.scaling_source_a, A),
                ),
            }
        else:
            workspace.scaling = None
            workspace.scaling_source_p = None
            workspace.scaling_source_a = None
            workspace.scaling_passes = 0


def _reconcile_workspace_constraint_types(workspace, constraint_types):
    previous = workspace.constraint_types
    if previous is not None and not torch.equal(previous, constraint_types):
        # Pinned l/u updates retain the current iterate and scalar rho.  A row
        # type change only rebuilds the rho-dependent linear system.
        workspace.linear_solver.clear()
    workspace.constraint_types = constraint_types.detach().clone()


def _rows_reordered(previous, current):
    if (
        previous is None
        or previous.shape != current.shape
        or previous.shape[0] < 2
        or torch.equal(previous, current)
    ):
        return False
    _, inverse = torch.unique(
        torch.cat((previous, current), dim=0),
        dim=0,
        return_inverse=True,
    )
    rows = previous.shape[0]
    previous_ids = inverse[:rows]
    current_ids = inverse[rows:]
    return bool(
        torch.equal(torch.sort(previous_ids).values, torch.sort(current_ids).values)
        and not torch.equal(previous_ids, current_ids)
    )


def _scale_data(P, q, A, passes):
    """``scaling.c::scale_data`` on the given (unscaled) ``P``, ``q``, ``A``.

    Returns ``D``, ``E``, ``c`` and the scaled ``P`` (upper triangle mirrored),
    ``A`` and ``q``.  The bounds are scaled by the caller (``l * E``, ``u * E``).
    """

    D = torch.ones(P.shape[0], device=P.device, dtype=P.dtype)
    E = torch.ones(A.shape[0], device=A.device, dtype=A.dtype)
    cost = torch.ones((), device=P.device, dtype=P.dtype)
    P_work = P.clone()
    A_work = A.clone()
    q_work = q.clone()
    for _ in range(passes):
        p_upper = torch.triu(P_work)
        p_norm = torch.amax(torch.abs(p_upper), dim=0)
        a_col = (
            torch.amax(torch.abs(A_work), dim=0)
            if A_work.shape[0]
            else torch.zeros_like(p_norm)
        )
        a_row = (
            torch.amax(torch.abs(A_work), dim=1)
            if A_work.shape[0]
            else torch.empty(0, device=A.device, dtype=A.dtype)
        )
        d_norm = _limit_scaling(torch.maximum(p_norm, a_col))
        e_norm = _limit_scaling(a_row)
        d_step = _ruiz_sqrt(d_norm).reciprocal()
        e_step = _ruiz_sqrt(e_norm).reciprocal()
        P_work = d_step[:, None] * P_work * d_step[None, :]
        A_work = e_step[:, None] * A_work * d_step[None, :]
        q_work = d_step * q_work
        D = D * d_step
        E = E * e_step

        cost_column_norms = torch.amax(torch.abs(torch.triu(P_work)), dim=0)
        average_p_norm = _cost_column_norm_sum(cost_column_norms) / P.shape[0]
        q_norm = _limit_scaling(_inf_norm(q_work))
        cost_step = _limit_scaling(torch.maximum(average_p_norm, q_norm)).reciprocal()
        P_work = cost_step * P_work
        q_work = cost_step * q_work
        cost = cost * cost_step
    # C stores only P's upper triangle; mirror those scaled entries.
    P_work = torch.triu(P_work) + torch.triu(P_work, diagonal=1).T
    return D, E, cost, P_work, A_work, q_work


def _scaling_for_problem(workspace, P, q, A, settings):
    passes = int(settings.get("scaling", 0))
    if passes <= 0:
        return {
            "D": torch.ones(P.shape[0], device=P.device, dtype=P.dtype),
            "E": torch.ones(A.shape[0], device=A.device, dtype=A.dtype),
            "cost": torch.ones((), device=P.device, dtype=P.dtype),
            "passes": 0,
        }
    previous = workspace.scaling
    if (
        previous is not None
        and workspace.scaling_passes == passes
        and previous.get("matrix_update") is not None
    ):
        # osqp_update_data_mat: unscale_data, insert the new values, scale_data.
        p_changed, a_changed = previous["matrix_update"]
        D_old, E_old, c_old = previous["D"], previous["E"], previous["cost"]
        Dinv, Einv, cinv = D_old.reciprocal(), E_old.reciprocal(), c_old.reciprocal()
        P_unscaled = ((cinv * previous["scaled_P"]) * Dinv[:, None]) * Dinv[None, :]
        q_unscaled = (previous["scaled_q"] * cinv) * Dinv
        A_unscaled = (Einv[:, None] * previous["scaled_A"]) * Dinv[None, :]
        P_new = P if p_changed else P_unscaled
        A_new = A if a_changed else A_unscaled
        D, E, cost, P_work, A_work, q_work = _scale_data(P_new, q_unscaled, A_new, passes)
        scaling = {
            "D": D,
            "E": E,
            "cost": cost,
            "passes": passes,
            "scaled_P": P_work,
            "scaled_A": A_work,
            "scaled_q": q_work,
            "source_q": previous.get("source_q"),
        }
        if previous.get("scaled_l") is not None:
            # unscale_data: l * Einv, u * Einv; scale_data: l * E, u * E.
            scaling["scaled_l"] = (previous["scaled_l"] * Einv) * E
            scaling["scaled_u"] = (previous["scaled_u"] * Einv) * E
            scaling["source_l"] = previous["source_l"]
            scaling["source_u"] = previous["source_u"]
        workspace.scaling = scaling
        workspace.scaling_source_p = P.detach().clone()
        workspace.scaling_source_a = A.detach().clone()
        previous = scaling
    if previous is not None and workspace.scaling_passes == passes:
        source_q = previous.get("source_q")
        if source_q is not None and torch.equal(source_q, q):
            # No new q: osqp_update_data_vec leaves work->data->q untouched.
            return previous
        # osqp_update_data_vec(q_new): q <- (q_new * D) * c.
        workspace.scaling = {
            **previous,
            "scaled_q": (q * previous["D"]) * previous["cost"],
            "source_q": q.detach().clone(),
        }
        return workspace.scaling
    D, E, cost, P_work, A_work, q_work = _scale_data(P, q, A, passes)
    scaling = {
        "D": D,
        "E": E,
        "cost": cost,
        "passes": passes,
        "scaled_P": P_work,
        "scaled_A": A_work,
        "scaled_q": q_work,
        "source_q": q.detach().clone(),
    }
    workspace.scaling = scaling
    workspace.scaling_source_p = P.detach().clone()
    workspace.scaling_source_a = A.detach().clone()
    workspace.scaling_passes = passes
    return scaling


def _cost_column_norm_sum(column_norms):
    if column_norms.dtype == torch.float64:
        return _seq_sum(column_norms)
    return torch.sum(column_norms)


def _c_sqrt(value):
    """``c_sqrt``: the correctly rounded IEEE square root used by the C build.

    ``torch.sqrt`` is not correctly rounded for every float64 input on CPU, so
    the scalar ``math.sqrt`` (the same operation as C ``sqrt``) is applied to
    each entry there.
    """

    if value.device.type == "cpu" and value.dtype == torch.float64:
        flat = [math.sqrt(entry) for entry in value.reshape(-1).tolist()]
        return value.new_tensor(flat).reshape(value.shape)
    return torch.sqrt(value)


def _ruiz_sqrt(norm):
    # scaling.c takes the square root first, then rounds the reciprocal separately.
    return _c_sqrt(norm)


def _scale_problem(P, q, A, l, u, scaling):
    """Scaled work data; ``scaling["bounds_set"]`` says whether ``l``/``u`` were (re)set.

    ``osqp_setup`` and ``osqp_update_data_vec(l, u)`` store ``l * E`` and
    ``u * E`` and refresh the constraint types.  Without a bound update the
    stored scaled bounds stay as they are, including after
    ``osqp_update_data_mat`` rescaled them.
    """

    if scaling["passes"] == 0:
        scaling["bounds_set"] = True
        return P, q, A, l, u
    E = scaling["E"]
    bounds_set = not (
        scaling.get("scaled_l") is not None
        and torch.equal(scaling["source_l"], l)
        and torch.equal(scaling["source_u"], u)
    )
    if bounds_set:
        scaling["scaled_l"] = E * l
        scaling["scaled_u"] = E * u
        scaling["source_l"] = l.detach().clone()
        scaling["source_u"] = u.detach().clone()
    scaling.pop("matrix_update", None)
    scaling["bounds_set"] = bounds_set
    return (
        scaling["scaled_P"],
        scaling["scaled_q"],
        scaling["scaled_A"],
        scaling["scaled_l"],
        scaling["scaled_u"],
    )


def _initial_scaled_state(workspace, n, m, device, dtype, scaling, warm_start, A_s):
    zeros = (
        torch.zeros(n, device=device, dtype=dtype),
        torch.zeros(m, device=device, dtype=dtype),
        torch.zeros(m, device=device, dtype=dtype),
    )
    if not warm_start:
        # osqp_solve: if (!settings->warm_starting) osqp_cold_start(solver).
        return zeros
    retained = workspace.work_iterates
    if (
        isinstance(retained, dict)
        and retained.get("source") is workspace.state
        and all(
            torch.is_tensor(retained.get(key))
            and retained[key].device == device
            and retained[key].dtype == dtype
            for key in ("x", "z", "y")
        )
        and retained["x"].numel() == n
        and retained["z"].numel() == m
        and retained["y"].numel() == m
    ):
        # osqp_solve reuses work->x, work->z, work->y exactly as the previous
        # solve left them (scaled coordinates, no unscale/rescale round trip).
        return retained["x"].clone(), retained["z"].clone(), retained["y"].clone()
    if not isinstance(workspace.state, dict):
        return zeros
    # An externally supplied state is osqp_warm_start(x, y): scale x and y,
    # then set z = A x in scaled coordinates.
    x_o = _state_vector(workspace.state.get("x"), zeros[0])
    y_o = _state_vector(workspace.state.get("y"), zeros[2])
    D, E, cost = scaling["D"], scaling["E"], scaling["cost"]
    x = x_o * D.reciprocal()
    y = (y_o * E.reciprocal()) * cost
    return x, _matvec(A_s, x), y


def _state_vector(value, fallback):
    if not torch.is_tensor(value):
        return fallback
    value = value.detach().to(device=fallback.device, dtype=fallback.dtype).reshape(-1)
    if value.numel() != fallback.numel() or not bool(
        torch.all(torch.isfinite(value)).item()
    ):
        return fallback
    return value.clone()


def _unscale_state(x, z, y, scaling):
    D, E, cost = scaling["D"], scaling["E"], scaling["cost"]
    return D * x, z / E, (E * y) * cost.reciprocal()


def _equality_mask(l, u):
    return _constraint_types(l, u) == 1


def _constraint_types(l, u):
    lower_infinite = _lower_is_infinite(l)
    upper_infinite = _upper_is_infinite(u)
    loose = lower_infinite & upper_infinite
    equality = ~loose & ((u - l) < 1e-4)
    types = torch.zeros(l.shape, device=l.device, dtype=torch.int8)
    types[loose] = -1
    types[equality] = 1
    return types


def _rho_vector(rho_bar, equality, loose=None):
    rho_vec = torch.full(
        equality.shape,
        float(rho_bar.item()),
        device=rho_bar.device,
        dtype=rho_bar.dtype,
    )
    rho_vec[equality] *= 1000.0
    if loose is not None:
        rho_vec[loose] = 1e-6
    return rho_vec


def _adaptive_rho_update(
    rho_bar,
    equality,
    primal,
    dual,
    eps_primal,
    eps_dual,
    tolerance,
    *,
    loose=None,
    primal_normalizer=None,
    dual_normalizer=None,
):
    primal_scale = eps_primal if primal_normalizer is None else primal_normalizer
    dual_scale = eps_dual if dual_normalizer is None else dual_normalizer
    candidate = _compute_rho_estimate(
        rho_bar,
        primal,
        dual,
        primal_scale,
        dual_scale,
    )
    should_update = (candidate > tolerance * rho_bar) | (
        candidate < rho_bar / tolerance
    )
    changed = candidate != rho_bar
    if not bool((should_update & changed & torch.isfinite(candidate)).item()):
        return False, rho_bar, _rho_vector(rho_bar, equality, loose)
    return True, candidate, _rho_vector(candidate, equality, loose)


def _compute_rho_estimate(
    rho_bar,
    primal,
    dual,
    primal_normalizer,
    dual_normalizer,
):
    division_tolerance = torch.as_tensor(
        _division_tolerance(rho_bar), device=rho_bar.device, dtype=rho_bar.dtype
    )
    normalized_primal = primal / (primal_normalizer + division_tolerance)
    normalized_dual = dual / (dual_normalizer + division_tolerance)
    ratio = normalized_primal / normalized_dual
    estimate_raw = rho_bar * _c_sqrt(ratio)
    rho_min = torch.as_tensor(1e-6, device=rho_bar.device, dtype=rho_bar.dtype)
    estimate = torch.where(torch.isnan(estimate_raw), rho_min, estimate_raw)
    return estimate.clamp(min=rho_min, max=1e6)


def _residuals(
    P, q, A, x, z, y, eps_abs, eps_rel, *, scaling=None, return_normalizers=False
):
    Ax, Px, ATy = _matvec(A, x), _matvec(P, x), _rmatvec(A, y)
    primal_vector, dual_vector = Ax - z, q + Px + ATy
    Dinv = Einv = None
    cost_inverse = 1.0
    if scaling is not None:
        # auxil.c computes residuals in scaled coordinates, then rescales
        # the residual vectors. Unscaling x/y first changes cancellation.
        Dinv, Einv = scaling["D"].reciprocal(), scaling["E"].reciprocal()
        cost_inverse = scaling["cost"].reciprocal()
        primal_vector, dual_vector = Einv * primal_vector, Dinv * dual_vector
    primal = _inf_norm(primal_vector)
    dual = cost_inverse * _inf_norm(dual_vector)
    primal_normalizer, dual_normalizer = _termination_normalizers(
        Ax, Px, ATy, q, z, Dinv, Einv, cost_inverse
    )
    eps_primal = eps_abs + eps_rel * primal_normalizer
    eps_dual = eps_abs + eps_rel * dual_normalizer
    residuals = (primal, dual, eps_primal, eps_dual)
    if return_normalizers:
        return residuals, (primal_normalizer, dual_normalizer)
    return residuals


def _termination_normalizers(Ax, Px, ATy, q, z, Dinv, Einv, cost_inverse):
    if Dinv is not None:
        Ax, z = Einv * Ax, Einv * z
        Px, ATy, q = Dinv * Px, Dinv * ATy, Dinv * q
    primal_normalizer = torch.maximum(
        _inf_norm(Ax),
        _inf_norm(z),
    )
    dual_normalizer = cost_inverse * torch.maximum(
        torch.maximum(
            _inf_norm(Px),
            _inf_norm(ATy),
        ),
        _inf_norm(q),
    )
    return primal_normalizer, dual_normalizer


def _residual_normalizers(P, q, A, x, z, y):
    Ax, Px, ATy = _matvec(A, x), _matvec(P, x), _rmatvec(A, y)
    return (
        torch.maximum(_inf_norm(Ax), _inf_norm(z)),
        torch.maximum(torch.maximum(_inf_norm(Px), _inf_norm(ATy)), _inf_norm(q)),
    )


def _compute_obj_val_dual_gap(P, q, l, u, x, y, cost, eps_abs, eps_rel, *, Px=None):
    """Pinned auxil.c objective/gap calculation in scaled working coordinates.

    User objective values stay unscaled. Both gap coordinate systems are
    returned so termination can select the source-requested one.
    """
    quad_term = _dot(_matvec(P, x) if Px is None else Px, x)
    lin_term = _dot(q, x)
    lower_infinite = _lower_is_infinite(l)
    upper_infinite = _upper_is_infinite(u)
    projected = torch.where(upper_infinite, torch.clamp_max(y, 0.0), y)
    projected = torch.where(lower_infinite, torch.clamp_min(projected, 0.0), projected)
    projected = torch.where(lower_infinite & upper_infinite, 0.0, projected)
    deadzone = 1e-10 if y.dtype == torch.float32 else 1e-15
    projected = torch.where(projected.abs() < deadzone, 0.0, projected)
    # Mask unused IEEE-infinite bounds before multiplying, avoiding Inf * 0.
    # Finite OSQP sentinel boundaries retain the source's strict comparisons.
    positive = torch.clamp_min(projected, 0.0)
    negative = torch.clamp_max(projected, 0.0)
    sup_term = _dot(torch.where(positive == 0.0, 0.0, u), positive)
    sup_term = sup_term + _dot(torch.where(negative == 0.0, 0.0, l), negative)
    max_term = torch.maximum(torch.maximum(quad_term.abs(), lin_term.abs()), sup_term.abs())
    cinv = cost.reciprocal()
    scaled_gap = quad_term + lin_term + sup_term
    gap_normalizer = max_term * cinv
    return {
        "objective": (0.5 * quad_term + lin_term) * cinv,
        "dual_objective": (-0.5 * quad_term - sup_term) * cinv,
        "duality_gap": scaled_gap * cinv,
        "scaled_duality_gap": scaled_gap,
        "eps_duality_gap": eps_abs + eps_rel * gap_normalizer,
        "eps_scaled_duality_gap": eps_abs + eps_rel * max_term,
        "gap_normalizer": gap_normalizer,
        "scaled_gap_normalizer": max_term,
    }


def _termination_status(
    P,
    q,
    A,
    l,
    u,
    delta_x,
    delta_y,
    residuals,
    settings,
    *,
    approximate,
    objective_values,
    dual_bounds=None,
    primal_certificate_context=None,
    dual_certificate_context=None,
):
    # The caller supplies tolerances for this check's coordinate system and mode.
    primal, dual, eps_primal, eps_dual = residuals
    infinity = _osqp_infinity(primal)
    if bool(((primal > infinity) | (dual > infinity)).item()):
        return "non_convex", None, None

    multiplier = 10.0 if approximate else 1.0
    primal_satisfied = A.shape[0] == 0 or bool((primal < eps_primal).item())
    dual_satisfied = bool((dual < eps_dual).item())
    gap_key = (
        "scaled_duality_gap"
        if settings.get("scaled_termination", False)
        else "duality_gap"
    )
    gap_tolerance_key = (
        "eps_scaled_duality_gap"
        if settings.get("scaled_termination", False)
        else "eps_duality_gap"
    )
    gap_satisfied = not settings.get("check_dualgap", P.dtype == torch.float64) or bool(
        (objective_values[gap_key].abs() < objective_values[gap_tolerance_key]).item()
    )
    if primal_satisfied and dual_satisfied and gap_satisfied:
        return ("solved_inaccurate" if approximate else "solved"), None, None

    primal_certificate = None
    if not primal_satisfied:
        tolerance = multiplier * float(settings.get("eps_prim_inf", 1e-4))
        if primal_certificate_context is None:
            primal_certificate = _primal_infeasibility_certificate(
                A, l, u, delta_y, tolerance
            )
        else:
            (
                primal_A,
                primal_l,
                primal_u,
                primal_delta_y,
                norm_scale,
                at_scale,
                certificate_scale,
            ) = primal_certificate_context
            primal_certificate = _primal_infeasibility_certificate(
                primal_A,
                primal_l,
                primal_u,
                primal_delta_y,
                tolerance,
                norm_scale=norm_scale,
                at_scale=at_scale,
                certificate_scale=certificate_scale,
            )
    if primal_certificate is not None:
        status = "primal_infeasible_inaccurate" if approximate else "primal_infeasible"
        return status, primal_certificate, None

    dual_certificate = None
    if not dual_satisfied:
        # OSQP checks original-unit directions against scaled work-data bounds.
        tolerance = multiplier * float(settings.get("eps_dual_inf", 1e-4))
        if dual_certificate_context is None:
            dual_l, dual_u = (l, u) if dual_bounds is None else dual_bounds
            dual_certificate = _dual_infeasibility_certificate(
                P, q, A, dual_l, dual_u, delta_x, tolerance
            )
        else:
            (
                dual_P,
                dual_q,
                dual_A,
                dual_l,
                dual_u,
                dual_delta_x,
                norm_scale,
                p_scale,
                a_scale,
                cost_scale,
                certificate_scale,
            ) = dual_certificate_context
            dual_certificate = _dual_infeasibility_certificate(
                dual_P,
                dual_q,
                dual_A,
                dual_l,
                dual_u,
                dual_delta_x,
                tolerance,
                norm_scale=norm_scale,
                p_scale=p_scale,
                a_scale=a_scale,
                cost_scale=cost_scale,
                certificate_scale=certificate_scale,
            )
    if dual_certificate is not None:
        status = "dual_infeasible_inaccurate" if approximate else "dual_infeasible"
        return status, None, dual_certificate
    return None


def _project_polar_reccone(delta_y, l, u):
    """``OSQPVectorf_project_polar_reccone`` with ``OSQP_INFTY * OSQP_MIN_SCALING``."""

    projected = delta_y.clone()
    lower_infinite = _lower_is_infinite(l)
    upper_infinite = _upper_is_infinite(u)
    projected[lower_infinite & upper_infinite] = 0.0
    upper_only_infinite = upper_infinite & ~lower_infinite
    lower_only_infinite = lower_infinite & ~upper_infinite
    projected[upper_only_infinite] = torch.clamp_max(projected[upper_only_infinite], 0.0)
    projected[lower_only_infinite] = torch.clamp_min(projected[lower_only_infinite], 0.0)
    return projected


def _primal_infeasibility_certificate(
    A,
    l,
    u,
    delta_y,
    tolerance,
    *,
    norm_scale=None,
    at_scale=None,
    certificate_scale=None,
):
    projected = delta_y.clone()
    lower_infinite = _lower_is_infinite(l)
    upper_infinite = _upper_is_infinite(u)
    projected[lower_infinite & upper_infinite] = 0.0
    upper_only_infinite = upper_infinite & ~lower_infinite
    lower_only_infinite = lower_infinite & ~upper_infinite
    projected[upper_only_infinite] = torch.minimum(
        projected[upper_only_infinite], torch.zeros_like(projected[upper_only_infinite])
    )
    projected[lower_only_infinite] = torch.maximum(
        projected[lower_only_infinite], torch.zeros_like(projected[lower_only_infinite])
    )

    norm_direction = projected if norm_scale is None else norm_scale * projected
    norm = _inf_norm(norm_direction)
    if not bool((norm > _division_tolerance(projected)).item()):
        return None
    positive = torch.clamp_min(projected, 0.0)
    negative = torch.clamp_max(projected, 0.0)
    support = torch.zeros((), device=projected.device, dtype=projected.dtype)
    positive_support = (~upper_infinite) & (positive > 0.0)
    negative_support = (~lower_infinite) & (negative < 0.0)
    if bool(torch.any(positive_support).item()):
        support = support + _dot(
            u[positive_support], positive[positive_support]
        )
    if bool(torch.any(negative_support).item()):
        support = support + _dot(
            l[negative_support], negative[negative_support]
        )
    if not bool((support < 0.0).item()):
        return None
    at_delta_y = _rmatvec(A, projected)
    if at_scale is not None:
        at_delta_y = at_scale * at_delta_y
    if not bool((_inf_norm(at_delta_y) < tolerance * norm).item()):
        return None
    normalized = projected if certificate_scale is None else certificate_scale * projected
    return normalized * norm.reciprocal()


def _dual_infeasibility_certificate(
    P,
    q,
    A,
    l,
    u,
    delta_x,
    tolerance,
    *,
    norm_scale=None,
    p_scale=None,
    a_scale=None,
    cost_scale=None,
    certificate_scale=None,
):
    norm_direction = delta_x if norm_scale is None else norm_scale * delta_x
    norm = _inf_norm(norm_direction)
    if not bool((norm > _division_tolerance(delta_x)).item()):
        return None
    if not bool((_dot(q, delta_x) < 0.0).item()):
        return None
    p_delta_x = _matvec(P, delta_x)
    if p_scale is not None:
        p_delta_x = p_scale * p_delta_x
    if cost_scale is None:
        p_tolerance = tolerance * norm
    else:
        p_tolerance = (cost_scale * tolerance) * norm
    if not bool((_inf_norm(p_delta_x) < p_tolerance).item()):
        return None

    direction = _matvec(A, delta_x)
    if a_scale is not None:
        direction = a_scale * direction
    cone_tolerance = tolerance * norm
    upper_finite = u < _osqp_infinity(u) * 1e-4
    lower_finite = l > -(_osqp_infinity(l) * 1e-4)
    upper_violation = upper_finite & (direction > cone_tolerance)
    lower_violation = lower_finite & (direction < -cone_tolerance)
    if bool(torch.any(upper_violation | lower_violation).item()):
        return None
    normalized = delta_x if certificate_scale is None else certificate_scale * delta_x
    return normalized * norm.reciprocal()


def _seq_sum(values, dim=0, start=None):
    """Sum along ``dim`` in index order, one rounded addition per term.

    The C loops (``csc_Axpy``, ``csc_Atxpy``, ``csc_Axpy_sym_triu``,
    ``OSQPVectorf_dot_prod``) accumulate ``acc += term`` from ``acc = 0.0`` (or
    from the existing entry when ``beta == 1``).  ``torch.cumsum`` performs
    exactly that recurrence for CPU float64; other devices and dtypes use the
    explicit recurrence because their ``cumsum`` does not.
    """

    first = (
        torch.zeros_like(values.select(dim, 0)) if values.shape[dim] else None
    ) if start is None else start
    if values.shape[dim] == 0:
        if start is not None:
            return start.clone()
        shape = list(values.shape)
        del shape[dim]
        return torch.zeros(shape, device=values.device, dtype=values.dtype)
    if values.device.type == "cpu" and values.dtype == torch.float64:
        stacked = torch.cat((first.unsqueeze(dim), values), dim=dim)
        return torch.cumsum(stacked, dim=dim).select(dim, -1)
    total = first
    for index in range(values.shape[dim]):
        total = total + values.select(dim, index)
    return total


def _matvec(M, x, start=None, subtract=False):
    """``OSQPMatrix_Axpy``: row ``r`` accumulates ``M[r, c] * x[c]`` for increasing ``c``.

    This is the accumulation order of ``csc_Axpy`` for a general matrix and of
    ``csc_Axpy_sym_triu`` for the symmetric ``P`` stored as its upper triangle.
    """

    products = M * x.unsqueeze(0)
    return _seq_sum(-products if subtract else products, dim=1, start=start)


def _rmatvec(A, y, start=None, subtract=False):
    """``csc_Atxpy``: column ``j`` accumulates ``A[k, j] * y[k]`` for increasing ``k``."""

    products = A * y.unsqueeze(1)
    return _seq_sum(-products if subtract else products, dim=0, start=start)


def _dot(a, b):
    """``OSQPVectorf_dot_prod``: ``dotprod += a[i] * b[i]`` in index order."""

    return _seq_sum(a * b, dim=0)


def _inf_norm(value):
    if value.numel() == 0:
        return torch.zeros((), device=value.device, dtype=value.dtype)
    return torch.linalg.vector_norm(value, ord=float("inf"))


def _limit_scaling(value):
    one = torch.ones_like(value)
    limited = torch.minimum(value, torch.full_like(value, 1e4))
    return torch.where(value < 1e-4, one, limited)


def _osqp_infinity(value):
    if value.dtype == torch.float32 and value.device.type == "cuda":
        return 1e17
    return 1e30


def _division_tolerance(value):
    return 1.0 / _osqp_infinity(value)


def _lower_is_infinite(l):
    return l < -(_osqp_infinity(l) * 1e-4)


def _upper_is_infinite(u):
    return u > _osqp_infinity(u) * 1e-4


def _default_polish_info(settings):
    enabled = bool(settings.get("polishing", False))
    return {
        "polishing": enabled,
        "polishing_success": False,
        "polishing_status": "not_run" if enabled else "disabled",
        "polishing_time_ms": 0.0,
        "polishing_active_constraints": 0,
        "polishing_refine_iter": int(settings.get("polish_refine_iter", 3)),
    }


def _polish_solution(
    P,
    q,
    A,
    l,
    u,
    x,
    z,
    y,
    settings,
    *,
    scaling=None,
    p_pattern=None,
    a_pattern=None,
):
    info = _default_polish_info(settings)
    started = _clock()
    equality = torch.isfinite(l) & torch.isfinite(u) & (l == u)
    lower_active = equality | (torch.isfinite(l) & ((z - l) < -y))
    upper_active = ~lower_active & torch.isfinite(u) & ((u - z) < y)
    active = lower_active | upper_active
    # form_Ared in pinned polish.c visits active rows in their original order.
    indices = torch.nonzero(active, as_tuple=False).reshape(-1)
    info["polishing_active_constraints"] = int(indices.numel())
    if indices.numel() == 0:
        info["polishing_status"] = "skipped_no_active_constraints"
        info["polishing_time_ms"] = (_clock() - started) * 1000
        return x, z, y, info

    A_active = A[indices]
    rhs_active = torch.where(
        lower_active[indices], l[indices], u[indices],
    )
    # The pinned direct backend uses sigma for both signed polish blocks.
    regularization = float(settings.get("sigma", 1e-6))
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
                    regularization * torch.eye(n, device=q.device, dtype=q.dtype),
                    torch.zeros((n, p), device=q.device, dtype=q.dtype),
                ),
                dim=1,
            ),
            torch.cat(
                (
                    torch.zeros((p, n), device=q.device, dtype=q.dtype),
                    -regularization * torch.eye(p, device=q.device, dtype=q.dtype),
                ),
                dim=1,
            ),
        ),
        dim=0,
    )
    K_regularized = K_exact + regularizer
    rhs = torch.cat((-q, rhs_active))
    # polish.c returns the linear-system exit flag and osqp_solve then exits
    # with "Failed polishing" before store_solution; errors propagate here too.
    if settings.get("use_sparse", False):
        polish_solver = SparseDirectSolver()
        polish_solver.factorize(K_regularized)
    else:
        # init_linsys_solver_qdldl(..., polishing=1): form_KKT(P, Ared, sigma,
        # NULL, sigma), AMD, LDL_factor with the same positive-pivot check.
        polish_solver = QDLDLSolver()
        polish_solver.factorize(
            K_regularized,
            P=P,
            A=A_active,
            sigma=regularization,
            rho_inv_vec=None,
            p_pattern=P.ne(0) if p_pattern is None else p_pattern,
            a_pattern=(A.ne(0) if a_pattern is None else a_pattern)[indices],
        )
        if polish_solver.positive_d_count < n:
            raise TorchLinearSolveError(
                "Error in KKT matrix LDL factorization when computing the "
                "nonzero elements. The problem seems to be non-convex."
            )
    polished, _ = polish_solver.solve(
        rhs,
        calculate_residual=bool(settings.get("check_linear_residual", False)),
    )
    for _ in range(int(settings.get("polish_refine_iter", 3))):
        # polish.c::iterative_refinement subtracts the two top blocks in
        # sequence. Combining them changes cancellation and rounding.
        refinement_rhs = torch.cat(
            (
                # OSQPMatrix_Axpy / Atxpy with alpha = -1, beta = 1 subtract each
                # product from the running entry in storage order.
                _rmatvec(
                    A_active, polished[n:],
                    start=_matvec(P, polished[:n], start=rhs[:n], subtract=True),
                    subtract=True,
                ),
                _matvec(A_active, polished[:n], start=rhs[n:], subtract=True),
            )
        )
        correction, _ = polish_solver.solve(refinement_rhs)
        polished = polished + correction
    info["polishing_regularized"] = True

    x_candidate = polished[:n]
    y_candidate = torch.zeros_like(y)
    y_candidate[indices] = polished[n:]
    z_unprojected = _matvec(A, x_candidate)
    shifted = y_candidate + z_unprojected
    z_candidate = torch.maximum(torch.minimum(shifted, u), l)
    y_candidate = shifted - z_candidate
    residual_scaling = (
        None if bool(settings.get("scaled_termination", False)) else scaling
    )
    old = _residuals(
        P, q, A, x, z, y,
        float(settings["eps_abs"]),
        float(settings["eps_rel"]),
        scaling=residual_scaling,
    )
    new = _residuals(
        P, q, A, x_candidate, z_candidate, y_candidate,
        float(settings["eps_abs"]),
        float(settings["eps_rel"]),
        scaling=residual_scaling,
    )
    candidate_objective = _compute_obj_val_dual_gap(
        P, q, l, u, x_candidate, y_candidate,
        q.new_tensor(1.0) if scaling is None else scaling["cost"],
        float(settings["eps_abs"]), float(settings["eps_rel"]),
    )
    # C writes the shared diagnostic before deciding whether to accept polish.
    info["rel_kkt_error"] = float(_relative_kkt_error(
        new[0], new[1], candidate_objective["duality_gap"]
    ).item())
    tiny = torch.as_tensor(torch.finfo(q.dtype).tiny, device=q.device, dtype=q.dtype)
    old_score = torch.maximum(
        old[0] / old[2].clamp_min(tiny), old[1] / old[3].clamp_min(tiny)
    )
    new_score = torch.maximum(
        new[0] / new[2].clamp_min(tiny), new[1] / new[3].clamp_min(tiny)
    )
    accepted = (
        ((new[0] < old[0]) & (new[1] < old[1]))
        | ((new[0] < old[0]) & (old[1] < 1e-10))
        | ((new[1] < old[1]) & (old[0] < 1e-10))
    )
    if not bool(accepted.item()):
        info.update(
            {
                "polishing_status": "rejected_no_improvement",
                "polishing_old_score": float(old_score.item()),
                "polishing_new_score": float(new_score.item()),
                "polishing_time_ms": (_clock() - started) * 1000,
            }
        )
        return x, z, y, info
    info.update(
        {
            "polishing_success": True,
            "polishing_status": "accepted",
            "polishing_old_score": float(old_score.item()),
            "polishing_new_score": float(new_score.item()),
            "polishing_time_ms": (_clock() - started) * 1000,
        }
    )
    return x_candidate, z_candidate, y_candidate, info
