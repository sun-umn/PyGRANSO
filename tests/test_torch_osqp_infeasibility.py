import numpy as np
import pytest
import torch

from pygranso.private import torchOSQP as torch_osqp_module
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchOSQP import (
    _dual_infeasibility_certificate,
    _primal_infeasibility_certificate,
    _termination_status,
    solve_torch_osqp_direct,
)


def _settings(**updates):
    settings = {
        "rho": 0.1,
        "sigma": 1e-6,
        "alpha": 1.6,
        "max_iter": 1000,
        "eps_abs": 1e-6,
        "eps_rel": 1e-6,
        "eps_prim_inf": 1e-4,
        "eps_dual_inf": 1e-4,
        "check_termination": 1,
        "scaling": 0,
        "adaptive_rho": False,
        "rho_update_interval": 50,
        "rho_update_tolerance": 5.0,
        "warm_start": True,
        "polishing": False,
        "return_info": True,
    }
    settings.update(updates)
    return settings


def _pinned_sparse_random_dense(rows, columns, generator):
    size = int(round(0.01 * rows * columns))
    flat_indices = generator.choice(rows * columns, size=size, replace=False)
    row_indices, column_indices = np.unravel_index(
        flat_indices, (rows, columns), order="F"
    )
    values = generator.uniform(size=size)
    matrix = np.zeros((rows, columns), dtype=np.float64)
    matrix[row_indices, column_indices] = values
    return matrix


def _pinned_primal_dual_fixture():
    dtype = torch.float64
    return {
        "P": torch.diag(torch.tensor([1.0, 0.0], dtype=dtype)),
        "q": torch.tensor([1.0, -1.0], dtype=dtype),
        "A12": torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=dtype),
        "A34": torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=dtype),
        "l": torch.tensor([0.0, 1.0, 1.0], dtype=dtype),
        "u1": torch.tensor([5.0, 3.0, 3.0], dtype=dtype),
        "u2": torch.tensor([0.0, 3.0, 3.0], dtype=dtype),
        "u3": torch.tensor([2.0, 3.0, torch.inf], dtype=dtype),
        "u4": torch.tensor([0.0, 3.0, torch.inf], dtype=dtype),
    }


def _assert_primal_certificate(info, A, l, u):
    certificate = info["primal_infeasibility_certificate"]
    assert certificate is not None
    assert info["dual_infeasibility_certificate"] is None
    assert info["objective"] == A.new_tensor(1e30).item()
    assert not info["status_compatible"]
    assert torch.all(torch.isfinite(certificate))
    norm = torch.linalg.vector_norm(certificate, ord=float("inf"))
    assert torch.isclose(norm, torch.ones_like(norm))
    tolerance = 1e-4
    assert torch.linalg.vector_norm(A.T @ certificate, ord=float("inf")) < (
        tolerance + 1e-12
    )
    assert torch.all(certificate[~torch.isfinite(u)] <= tolerance)
    assert torch.all(certificate[~torch.isfinite(l)] >= -tolerance)
    finite_u = torch.where(torch.isfinite(u), u, torch.zeros_like(u))
    finite_l = torch.where(torch.isfinite(l), l, torch.zeros_like(l))
    support = torch.dot(finite_u, torch.clamp(certificate, min=0.0)) + torch.dot(
        finite_l, torch.clamp(certificate, max=0.0)
    )
    assert support < 0


def _assert_dual_certificate(info, P, q, A, l, u):
    certificate = info["dual_infeasibility_certificate"]
    assert certificate is not None
    assert info["primal_infeasibility_certificate"] is None
    assert info["objective"] == A.new_tensor(-1e30).item()
    assert not info["status_compatible"]
    assert torch.all(torch.isfinite(certificate))
    norm = torch.linalg.vector_norm(certificate, ord=float("inf"))
    assert torch.isclose(norm, torch.ones_like(norm))
    tolerance = 1e-4
    assert torch.linalg.vector_norm(P @ certificate, ord=float("inf")) < (
        tolerance + 1e-12
    )
    assert torch.dot(q, certificate) < 0
    direction = A @ certificate
    assert torch.all(direction[torch.isfinite(u)] <= tolerance)
    assert torch.all(direction[torch.isfinite(l)] >= -tolerance)


def test_primal_certificate_projects_one_sided_and_free_rows_without_nan():
    dtype = torch.float64
    A = torch.tensor([[1.0], [1.0], [9.0], [0.0]], dtype=dtype)
    l = torch.tensor([1.0, -torch.inf, -torch.inf, -torch.inf], dtype=dtype)
    u = torch.tensor([torch.inf, 0.0, torch.inf, -torch.inf], dtype=dtype)
    delta_y = torch.tensor([-2.0, 2.0, 7.0, -3.0], dtype=dtype)
    certificate = _primal_infeasibility_certificate(A, l, u, delta_y, 1e-4)
    assert certificate is not None
    assert torch.all(torch.isfinite(certificate))
    assert torch.allclose(certificate, torch.tensor([-1.0, 1.0, 0.0, 0.0], dtype=dtype))
    assert torch.linalg.vector_norm(A.T @ certificate, ord=float("inf")) == 0


def test_dual_certificate_checks_cost_psd_and_recession_cone():
    dtype = torch.float64
    P = torch.zeros((1, 1), dtype=dtype)
    q = torch.tensor([-1.0], dtype=dtype)
    A = torch.empty((0, 1), dtype=dtype)
    bounds = torch.empty(0, dtype=dtype)
    certificate = _dual_infeasibility_certificate(
        P, q, A, bounds, bounds, torch.tensor([3.0], dtype=dtype), 1e-4
    )
    assert certificate is not None
    assert torch.equal(certificate, torch.ones(1, dtype=dtype))

    bounded_A = torch.ones((1, 1), dtype=dtype)
    assert (
        _dual_infeasibility_certificate(
            P,
            q,
            bounded_A,
            torch.tensor([-torch.inf], dtype=dtype),
            torch.tensor([0.0], dtype=dtype),
            torch.tensor([1.0], dtype=dtype),
            1e-4,
        )
        is None
    )

    source_infinity_boundary = 1e26
    certificate = _dual_infeasibility_certificate(
        P,
        q,
        bounded_A,
        torch.tensor([-source_infinity_boundary], dtype=dtype),
        torch.tensor([source_infinity_boundary], dtype=dtype),
        torch.tensor([1.0], dtype=dtype),
        1e-4,
    )
    assert certificate is not None


def test_infeasibility_uses_direct_scaled_iterate_differences(monkeypatch):
    """Pinned auxil.c keeps ADMM increments instead of subtracting huge states."""
    dtype = torch.float64
    P = torch.zeros((1, 1), dtype=dtype)
    q = torch.zeros(1, dtype=dtype)
    A = torch.empty((0, 1), dtype=dtype)
    bounds = torch.empty(0, dtype=dtype)
    scale = torch.tensor([1e16], dtype=dtype)
    scaling = {
        "D": scale,
        "E": torch.empty(0, dtype=dtype),
        "cost": torch.ones((), dtype=dtype),
        "passes": 1,
        "scaled_P": P.clone(),
        "scaled_q": q.clone(),
        "scaled_A": A.clone(),
    }
    monkeypatch.setattr(
        torch_osqp_module,
        "_scaling_for_problem",
        lambda workspace, P, q, A, settings: scaling,
    )
    states = iter(
        (
            torch.tensor([2**45], dtype=dtype),
            torch.tensor([2**45 + 1], dtype=dtype),
        )
    )

    def fake_admm_update(x_tilde, x, z_tilde, z, y, rho_vec, l, u, alpha, **kwargs):
        return next(states), z, y, torch.zeros_like(y)

    monkeypatch.setattr(torch_osqp_module, "admm_vector_update", fake_admm_update)
    captured = []

    def capture_termination(*args, **kwargs):
        captured.append((args[5].detach().clone(), args[6].detach().clone()))
        return None

    monkeypatch.setattr(torch_osqp_module, "_termination_status", capture_termination)
    solve_torch_osqp_direct(
        P,
        q,
        A,
        bounds,
        bounds,
        _settings(
            scaling=1,
            max_iter=2,
            check_termination=2,
            warm_start=False,
        ),
        TorchOSQPWorkspace(),
    )

    assert captured
    expected = scale * torch.ones(1, dtype=dtype)
    assert torch.equal(captured[0][0], expected)


def test_pinned_primal_dual_feasible_fixture_observables():
    problem = _pinned_primal_dual_fixture()
    solution, info = solve_torch_osqp_direct(
        problem["P"],
        problem["q"],
        problem["A12"],
        problem["l"],
        problem["u1"],
        _settings(
            max_iter=4000,
            eps_abs=1e-3,
            eps_rel=1e-3,
            check_termination=25,
            adaptive_rho=True,
            polishing=True,
            return_state=True,
        ),
        TorchOSQPWorkspace(),
    )

    assert info["status"] == "solved"
    assert info["status_val"] == 1
    assert torch.allclose(
        solution.reshape(-1), torch.tensor([1.0, 3.0], dtype=torch.float64)
    )
    assert torch.allclose(
        info["state"]["y"], torch.tensor([0.0, -2.0, 1.0], dtype=torch.float64)
    )
    assert info["objective"] == -1.5
    assert info["polishing_status"] == "accepted"


def test_pinned_primal_dual_infeasible_fixture_statuses_and_certificates():
    problem = _pinned_primal_dual_fixture()
    settings = _settings(
        max_iter=4000,
        eps_abs=1e-3,
        eps_rel=1e-3,
        check_termination=25,
        adaptive_rho=True,
        polishing=True,
    )
    cases = (
        ("primal", problem["A12"], problem["u2"]),
        ("dual", problem["A34"], problem["u3"]),
        ("either", problem["A34"], problem["u4"]),
    )

    for expected_family, A, u in cases:
        solution, info = solve_torch_osqp_direct(
            problem["P"],
            problem["q"],
            A,
            problem["l"],
            u,
            settings,
            TorchOSQPWorkspace(),
        )

        assert torch.all(solution == 0x7FC00000)
        assert info["polishing_status"] == "skipped_unsolved"
        if expected_family == "primal":
            assert info["status"] == "primal_infeasible"
            assert info["status_val"] == 3
            _assert_primal_certificate(info, A, problem["l"], u)
        elif expected_family == "dual":
            assert info["status"] == "dual_infeasible"
            assert info["status_val"] == 5
            _assert_dual_certificate(
                info, problem["P"], problem["q"], A, problem["l"], u
            )
        else:
            assert info["status"] in {"primal_infeasible", "dual_infeasible"}
            assert info["status_val"] in {3, 5}
            if info["status"] == "primal_infeasible":
                _assert_primal_certificate(info, A, problem["l"], u)
            else:
                _assert_dual_certificate(
                    info, problem["P"], problem["q"], A, problem["l"], u
                )


def test_pinned_random_primal_infeasibility_fixture_and_certificate():
    generator = np.random.Generator(np.random.PCG64(2))
    n, m = 50, 150
    Pt = _pinned_sparse_random_dense(n, n, generator)
    P = Pt.T @ Pt + np.eye(n)
    q = generator.standard_normal(n)
    A = _pinned_sparse_random_dense(m, n, generator)
    u = 3.0 + generator.standard_normal(m)
    l = -3.0 + generator.standard_normal(m)
    A[n // 2, :] = A[n // 2 + 1, :]
    l[n // 2] = u[n // 2 + 1] + 10.0 * generator.random()
    u[n // 2] = l[n // 2] + 0.5
    P, q, A, l, u = (torch.from_numpy(value) for value in (P, q, A, l, u))

    structural_certificate = torch.zeros(m, dtype=torch.float64)
    structural_certificate[n // 2] = -1.0
    structural_certificate[n // 2 + 1] = 1.0
    assert torch.equal(A[n // 2], A[n // 2 + 1])
    assert torch.linalg.vector_norm(A.T @ structural_certificate, ord=float("inf")) == 0
    assert u[n // 2 + 1] - l[n // 2] < 0

    solution, info = solve_torch_osqp_direct(
        P,
        q,
        A,
        l,
        u,
        _settings(
            max_iter=4000,
            eps_abs=1e-3,
            eps_rel=1e-3,
            check_termination=25,
            adaptive_rho=True,
            warm_start=False,
            polishing=True,
        ),
        TorchOSQPWorkspace(),
    )

    assert torch.all(solution == 0x7FC00000)
    assert info["status"] == "primal_infeasible"
    assert info["status_val"] == 3
    assert info["polishing_status"] == "skipped_unsolved"
    _assert_primal_certificate(info, A, l, u)


def test_termination_uses_strict_exact_and_tenfold_inaccurate_thresholds():
    dtype = torch.float64
    P = torch.zeros((1, 1), dtype=dtype)
    q = torch.zeros(1, dtype=dtype)
    A = torch.ones((1, 1), dtype=dtype)
    l = -torch.ones(1, dtype=dtype)
    u = torch.ones(1, dtype=dtype)
    zero = torch.zeros(1, dtype=dtype)
    one = torch.ones((), dtype=dtype)
    exact_boundary = (one, one, one, one)
    objective_values = {"duality_gap": torch.zeros_like(one), "eps_duality_gap": one}
    assert (
        _termination_status(
            P,
            q,
            A,
            l,
            u,
            zero,
            zero,
            exact_boundary,
            _settings(),
            approximate=False,
            objective_values=objective_values,
        )
        is None
    )
    below = torch.nextafter(one, torch.zeros_like(one))
    exact = _termination_status(
        P,
        q,
        A,
        l,
        u,
        zero,
        zero,
        (below, below, one, one),
        _settings(),
        approximate=False,
        objective_values=objective_values,
    )
    assert exact[0] == "solved"
    inaccurate = _termination_status(
        P,
        q,
        A,
        l,
        u,
        zero,
        zero,
        (5.0 * one, 5.0 * one, 10.0 * one, 10.0 * one),
        _settings(),
        approximate=True,
        objective_values=objective_values,
    )
    assert inaccurate[0] == "solved_inaccurate"
    assert inaccurate[1:] == (None, None)


@pytest.mark.parametrize("approximate", [False, True])
@pytest.mark.parametrize("scaled_termination", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float64])
def test_duality_gap_strict_thresholds_and_nonfinite_no_false_status(
    approximate, scaled_termination, device, dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    zero = torch.zeros(1, device=device, dtype=dtype)
    one = torch.ones((), device=device, dtype=dtype)
    threshold = one * (10 if approximate else 1)
    below = torch.nextafter(threshold, torch.zeros_like(threshold))
    above = torch.nextafter(threshold, threshold * 2)
    candidates = [
        (below, True),
        (threshold, False),
        (above, False),
        (one * float("inf"), False),
        (one * float("nan"), False),
    ]
    for gap, accepted in candidates:
        for sign in (-1, 1):
            result = _termination_status(
                zero.reshape(1, 1),
                zero,
                zero.reshape(1, 1),
                -one.reshape(1),
                one.reshape(1),
                zero,
                zero,
                (zero[0], zero[0], one, one),
                _settings(check_dualgap=True, scaled_termination=scaled_termination),
                approximate=approximate,
                objective_values={
                    "duality_gap": sign * gap if not scaled_termination else zero[0],
                    "eps_duality_gap": threshold,
                    "scaled_duality_gap": sign * gap if scaled_termination else zero[0],
                    "eps_scaled_duality_gap": threshold,
                },
            )
            expected = "solved_inaccurate" if approximate else "solved"
            assert result == ((expected, None, None) if accepted else None)


@pytest.mark.parametrize(
    "iterations,status",
    [(5, "max_iter_reached"), (12, "solved_inaccurate"), (13, "solved")],
)
def test_final_duality_gap_checks_with_periodic_termination_disabled(
    iterations, status
):
    empty = torch.empty(0, dtype=torch.float64)
    _, info = solve_torch_osqp_direct(
        torch.tensor([[1e-6]], dtype=torch.float64),
        torch.tensor([-1.0], dtype=torch.float64),
        empty.reshape(0, 1),
        empty,
        empty,
        _settings(
            max_iter=iterations,
            eps_abs=1e-3,
            eps_rel=0,
            eps_dual_inf=1e-12,
            check_termination=0,
            check_dualgap=True,
            polishing=True,
        ),
    )
    assert info["status"] == status
    assert info["admm_iterations"] == iterations
    assert info["primal_infeasibility_certificate"] is None
    assert info["dual_infeasibility_certificate"] is None
    if status != "solved":
        assert info["polishing_status"] == "skipped_unsolved"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float64])
@pytest.mark.parametrize("scaled_termination", [False, True])
def test_scaled_termination_returns_source_coordinate_certificates(
    device, dtype, scaled_termination
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    settings = _settings(
        scaling=10,
        scaled_termination=scaled_termination,
        check_termination=1,
        adaptive_rho=False,
        polishing=False,
    )
    P = torch.zeros((1, 1), device=device, dtype=dtype)
    q = torch.zeros(1, device=device, dtype=dtype)
    A = torch.tensor([[1e-3], [1e3]], device=device, dtype=dtype)
    l = torch.tensor([1e-3, -torch.inf], device=device, dtype=dtype)
    u = torch.tensor([torch.inf, 0.0], device=device, dtype=dtype)
    _, primal_info = solve_torch_osqp_direct(P, q, A, l, u, settings)
    expected_primal = (
        [-1.0, 0.9866695359287845]
        if scaled_termination
        else [-1.0, 1.0963698663860615e-6]
    )
    assert primal_info["status"] == "primal_infeasible"
    assert primal_info["admm_iterations"] == (22 if scaled_termination else 8)
    assert primal_info["dual_inf_cert"] is None
    assert torch.allclose(
        primal_info["prim_inf_cert"],
        torch.tensor(expected_primal, device=device, dtype=dtype),
        rtol=2e-5,
        atol=2e-7,
    )

    P = torch.zeros((2, 2), device=device, dtype=dtype)
    q = -torch.ones(2, device=device, dtype=dtype)
    A = torch.diag(torch.tensor([1e-3, 1e3], device=device, dtype=dtype))
    l = torch.full((2,), -torch.inf, device=device, dtype=dtype)
    u = torch.full((2,), torch.inf, device=device, dtype=dtype)
    _, dual_info = solve_torch_osqp_direct(P, q, A, l, u, settings)
    expected_dual = [1.0, 1e-3 if scaled_termination else 1e-6]
    assert dual_info["status"] == "dual_infeasible"
    assert dual_info["admm_iterations"] == 1
    assert dual_info["prim_inf_cert"] is None
    assert torch.allclose(
        dual_info["dual_inf_cert"],
        torch.tensor(expected_dual, device=device, dtype=dtype),
        rtol=2e-5,
        atol=2e-8,
    )


def test_divergent_residual_predicate_is_nonconvex_not_infeasible():
    dtype = torch.float64
    zero = torch.zeros(1, dtype=dtype)
    result = _termination_status(
        torch.zeros((1, 1), dtype=dtype),
        zero,
        torch.ones((1, 1), dtype=dtype),
        -torch.ones(1, dtype=dtype),
        torch.ones(1, dtype=dtype),
        zero,
        zero,
        (
            torch.tensor(float("inf"), dtype=dtype),
            torch.ones((), dtype=dtype),
            torch.ones((), dtype=dtype),
            torch.ones((), dtype=dtype),
        ),
        _settings(),
        approximate=False,
        objective_values={"duality_gap": torch.full((), torch.nan, dtype=dtype)},
    )
    assert result == ("non_convex", None, None)


def test_infeasible_request_cold_starts_previous_valid_warm_state():
    dtype = torch.float64
    P = torch.ones((1, 1), dtype=dtype)
    q = torch.zeros(1, dtype=dtype)
    A = torch.ones((2, 1), dtype=dtype)
    workspace = TorchOSQPWorkspace()
    feasible_l = torch.tensor([0.0, -torch.inf], dtype=dtype)
    feasible_u = torch.tensor([torch.inf, 1.0], dtype=dtype)
    _, feasible_info = solve_torch_osqp_direct(
        P, q, A, feasible_l, feasible_u, _settings(), workspace
    )
    assert feasible_info["status_compatible"]
    assert workspace.work_iterates is not None

    solution, info = solve_torch_osqp_direct(
        P,
        q,
        A,
        torch.tensor([1.0, -torch.inf], dtype=dtype),
        torch.tensor([torch.inf, 0.0], dtype=dtype),
        _settings(),
        workspace,
    )
    assert info["status"] in {
        "primal_infeasible",
        "primal_infeasible_inaccurate",
    }
    assert torch.all(solution == 0x7FC00000)
    assert info["primal_infeasibility_certificate"] is not None
    assert info["dual_infeasibility_certificate"] is None
    assert workspace.state is None and workspace.work_iterates is None
    assert workspace.linear_solver.ready
    assert workspace.last_info["status"] == info["status"]


def test_m_zero_and_disabled_periodic_checks_still_run_final_check():
    dtype = torch.float64
    solution, info = solve_torch_osqp_direct(
        torch.ones((1, 1), dtype=dtype),
        torch.tensor([-2.0], dtype=dtype),
        torch.empty((0, 1), dtype=dtype),
        torch.empty(0, dtype=dtype),
        torch.empty(0, dtype=dtype),
        _settings(check_termination=0),
        TorchOSQPWorkspace(),
    )
    assert info["status"] in {"solved", "solved_inaccurate"}
    assert info["status_compatible"]
    assert torch.allclose(solution, torch.tensor([[2.0]], dtype=dtype), atol=1e-4)
    assert info["primal_residual"] == 0.0
