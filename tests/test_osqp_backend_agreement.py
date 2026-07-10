import pytest
import torch

from pygranso.private.osqpTorchAdapter import solve_osqp_torch_qp
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace


def _solve(H, f, A, b, lower, upper, algebra):
    settings = {
        "return_info": True,
        "eps_abs": 1e-8,
        "eps_rel": 1e-8,
        "scaling": 10,
        "adaptive_rho": True,
        "polishing": True,
    }
    return solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        lower,
        upper,
        torch.device("cpu"),
        True,
        {"algebra": algebra, "settings": settings},
        TorchOSQPWorkspace(),
    )


@pytest.mark.parametrize("with_equality", [False, True])
def test_builtin_and_torch_agree_on_observable_outcomes(with_equality):
    dtype = torch.float64
    H = torch.tensor([[4.0, 1.0], [1.0, 2.0]], dtype=dtype)
    f = torch.tensor([[-1.0], [-1.0]], dtype=dtype)
    lower = torch.zeros((2, 1), dtype=dtype)
    upper = torch.ones((2, 1), dtype=dtype)
    A = torch.tensor([[1.0, 1.0]], dtype=dtype) if with_equality else None
    b = torch.tensor(0.75, dtype=dtype) if with_equality else None
    torch_solution, torch_info = _solve(H, f, A, b, lower, upper, "torch")
    builtin_solution, builtin_info = _solve(H, f, A, b, lower, upper, "builtin")
    assert torch_info["status_compatible"]
    assert builtin_info["status_compatible"]
    assert torch.allclose(torch_solution, builtin_solution, atol=2e-6, rtol=2e-6)
    objective_scale = max(1.0, abs(builtin_info["objective"]))
    assert abs(torch_info["objective"] - builtin_info["objective"]) / objective_scale < 1e-7
