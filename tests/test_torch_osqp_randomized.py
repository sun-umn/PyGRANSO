import torch

from pygranso.private.osqpTorchAdapter import solve_osqp_torch_qp
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace


def test_seeded_feasible_convex_qps_match_builtin(random_seed):
    generator = torch.Generator().manual_seed(random_seed)
    dtype = torch.float64
    n = 2 + random_seed % 5
    R = torch.randn((n, n), generator=generator, dtype=dtype)
    H = R.T @ R + 1e-3 * torch.eye(n, dtype=dtype)
    f = torch.randn((n, 1), generator=generator, dtype=dtype)
    lower = -torch.ones((n, 1), dtype=dtype)
    upper = torch.ones((n, 1), dtype=dtype)
    settings = {
        "return_info": True,
        "polishing": True,
        "scaling": 10,
        "adaptive_rho": True,
    }
    torch_x, torch_info = solve_osqp_torch_qp(
        H, f, None, None, lower, upper, "cpu", True,
        {"algebra": "torch", "settings": settings}, TorchOSQPWorkspace()
    )
    builtin_x, builtin_info = solve_osqp_torch_qp(
        H, f, None, None, lower, upper, "cpu", True,
        {"algebra": "builtin", "settings": settings}, TorchOSQPWorkspace()
    )
    assert torch_info["status_compatible"]
    assert builtin_info["status_compatible"]
    assert torch.allclose(torch_x, builtin_x, atol=2e-5, rtol=2e-5)
