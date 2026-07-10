import torch

from pygranso.private.osqpTorchAdapter import solve_osqp_torch_qp
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace


def _solve(H, f, lower, upper):
    return solve_osqp_torch_qp(
        H,
        f,
        None,
        None,
        lower,
        upper,
        "cpu",
        True,
        {"algebra": "torch", "settings": {"return_info": True}},
        TorchOSQPWorkspace(),
    )


def _problem():
    dtype = torch.float64
    H = torch.tensor(
        [[4.0, 0.5, 0.0], [0.5, 2.0, 0.25], [0.0, 0.25, 1.0]],
        dtype=dtype,
    )
    f = torch.tensor([[-1.0], [0.5], [-2.0]], dtype=dtype)
    lower = -torch.ones((3, 1), dtype=dtype)
    upper = torch.ones((3, 1), dtype=dtype)
    return H, f, lower, upper


def test_positive_objective_scaling_preserves_minimizer():
    H, f, lower, upper = _problem()
    base, base_info = _solve(H, f, lower, upper)
    scaled, scaled_info = _solve(7.0 * H, 7.0 * f, lower, upper)
    assert base_info["status_compatible"] and scaled_info["status_compatible"]
    assert torch.allclose(base, scaled, atol=2e-8, rtol=2e-8)


def test_variable_permutation_commutes_with_the_solver():
    H, f, lower, upper = _problem()
    base, _ = _solve(H, f, lower, upper)
    permutation = torch.tensor([2, 0, 1])
    permuted, _ = _solve(
        H[permutation][:, permutation],
        f[permutation],
        lower[permutation],
        upper[permutation],
    )
    assert torch.allclose(permuted, base[permutation], atol=2e-8, rtol=2e-8)
