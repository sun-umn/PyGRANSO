import pytest
import torch

from pygranso.private import torchOSQP as torch_osqp_module
from pygranso.private.osqpTorchAdapter import solve_osqp_torch_qp
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace


def _bound_problem(dtype=torch.float64, device="cpu"):
    device = torch.device(device)
    H = torch.eye(2, device=device, dtype=dtype)
    f = torch.tensor([[-1.0], [-2.0]], device=device, dtype=dtype)
    lower = torch.zeros((2, 1), device=device, dtype=dtype)
    upper = torch.ones((2, 1), device=device, dtype=dtype)
    return H, f, None, None, lower, upper


def _solve(problem, *, settings=None, workspace=None, algebra="torch"):
    H, f, A, b, lower, upper = problem
    options = {"algebra": algebra, "settings": {"return_info": True}}
    options["settings"].update(settings or {})
    return solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        lower,
        upper,
        f.device,
        f.dtype == torch.float64,
        options,
        workspace,
    )


def test_bound_qp_device_dtype_shape_and_solution(floating_dtype, available_device):
    solution, info = _solve(_bound_problem(floating_dtype, available_device))
    tolerance = 3e-4 if floating_dtype == torch.float32 else 1e-7
    assert solution.shape == (2, 1)
    assert solution.device.type == available_device.type
    if available_device.index is not None:
        assert solution.device.index == available_device.index
    assert solution.dtype == floating_dtype
    assert torch.allclose(
        solution.reshape(-1),
        torch.tensor([1.0, 1.0], device=available_device, dtype=floating_dtype),
        atol=tolerance,
        rtol=tolerance,
    )
    assert info["status"] == "solved"
    assert info["primal_residual"] <= info["eps_primal"]
    assert info["dual_residual"] <= info["eps_dual"]


def test_equality_plus_bounds_and_infinite_bound():
    dtype = torch.float64
    H = torch.eye(2, dtype=dtype)
    f = torch.zeros((2, 1), dtype=dtype)
    A = torch.tensor([[1.0, 1.0]], dtype=dtype)
    b = torch.tensor(1.0, dtype=dtype)
    lower = torch.tensor([[-torch.inf], [0.0]], dtype=dtype)
    upper = torch.tensor([[torch.inf], [1.0]], dtype=dtype)
    solution, info = _solve((H, f, A, b, lower, upper))
    assert torch.allclose(solution.sum(), torch.tensor(1.0, dtype=dtype), atol=1e-7)
    assert info["status"] == "solved"


def test_warm_start_and_lu_reuse_for_vector_update():
    workspace = TorchOSQPWorkspace()
    problem = _bound_problem()
    _, first = _solve(problem, settings={"polishing": False}, workspace=workspace)
    changed = list(problem)
    changed[1] = torch.tensor([[-0.5], [-1.5]], dtype=torch.float64)
    _, second = _solve(tuple(changed), settings={"polishing": False}, workspace=workspace)
    assert first["factorizations_this_solve"] >= 1
    assert second["factorizations_this_solve"] == 0
    assert second["factorization_reused"]


def test_matrix_value_update_refactorizes_but_keeps_workspace_state():
    workspace = TorchOSQPWorkspace()
    problem = _bound_problem()
    _solve(problem, settings={"polishing": False}, workspace=workspace)
    changed = list(problem)
    changed[0] = 2.0 * changed[0]
    _solve(tuple(changed), settings={"polishing": False}, workspace=workspace)
    assert workspace.state is not None
    assert workspace.linear_solver.factorization_count >= 2


def test_scaling_adaptive_rho_and_polishing_are_reported():
    _, info = _solve(_bound_problem())
    assert info["scaling_applied"]
    assert info["scaling_passes"] == 10
    assert "rho_updates" in info
    assert info["polishing_status"] == "accepted"


def test_requested_rho_change_refactorizes():
    workspace = TorchOSQPWorkspace()
    problem = _bound_problem()
    _solve(problem, settings={"polishing": False, "rho": 0.1}, workspace=workspace)
    _, info = _solve(
        problem,
        settings={"polishing": False, "rho": 0.2},
        workspace=workspace,
    )
    assert info["factorizations_this_solve"] >= 1
    assert workspace.rho_setting == 0.2


def test_requested_sigma_change_refactorizes():
    workspace = TorchOSQPWorkspace()
    problem = _bound_problem()
    _solve(problem, settings={"polishing": False, "sigma": 1e-6}, workspace=workspace)
    _, info = _solve(
        problem,
        settings={"polishing": False, "sigma": 1e-5},
        workspace=workspace,
    )
    assert info["factorizations_this_solve"] >= 1
    assert not info["factorization_reused"]


def test_incompatible_requested_dtype_is_rejected():
    problem = _bound_problem(torch.float32)
    with pytest.raises(ValueError, match="requested precision"):
        solve_osqp_torch_qp(
            *problem,
            torch.device("cpu"),
            True,
            {"algebra": "torch"},
            TorchOSQPWorkspace(),
        )


def test_input_validation_rejects_invalid_problem():
    problem = list(_bound_problem())
    problem[0] = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="asymmetric"):
        _solve(tuple(problem))

    problem = list(_bound_problem())
    problem[4] = torch.ones((2, 1), dtype=torch.float64)
    problem[5] = torch.zeros((2, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match="lower bound"):
        _solve(tuple(problem))

    problem = list(_bound_problem())
    problem[0][0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        _solve(tuple(problem))


def test_near_symmetric_p_is_safely_symmetrized():
    problem = list(_bound_problem())
    perturbation = 10.0 * torch.finfo(torch.float64).eps
    problem[0][0, 1] = perturbation
    solution, info = _solve(tuple(problem))
    assert torch.all(torch.isfinite(solution))
    assert info["status_compatible"]


def test_optional_convexity_diagnostic_rejects_indefinite_p():
    problem = list(_bound_problem())
    problem[0] = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="positive semidefinite"):
        _solve(tuple(problem), settings={"check_convexity": True})


def test_requested_polishing_rejection_raises(monkeypatch):
    def rejected(P, q, A, l, u, x, z, y, settings):
        return x, z, y, {"polishing_status": "rejected_no_improvement"}

    monkeypatch.setattr(torch_osqp_module, "_polish_solution", rejected)
    with pytest.raises(RuntimeError, match="polishing failed"):
        _solve(_bound_problem())
