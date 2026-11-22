import math

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
        H,
        f,
        None,
        None,
        lower,
        upper,
        "cpu",
        True,
        {"algebra": "torch", "settings": settings},
        TorchOSQPWorkspace(),
    )
    builtin_x, builtin_info = solve_osqp_torch_qp(
        H,
        f,
        None,
        None,
        lower,
        upper,
        "cpu",
        True,
        {"algebra": "builtin", "settings": settings},
        TorchOSQPWorkspace(),
    )
    assert torch_info["status_compatible"]
    assert builtin_info["status_compatible"]
    assert torch.allclose(torch_x, builtin_x, atol=2e-5, rtol=2e-5)


def test_scaled_adaptive_rho_stress_seed_57_converges():
    seed = 57
    dtype = torch.float64
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n = 2 + seed % 7
    eigenvalues = torch.logspace(0, math.log10(5e7), n, dtype=dtype)
    basis, _ = torch.linalg.qr(
        torch.randn((n, n), generator=generator, dtype=dtype)
    )
    H = basis @ torch.diag(eigenvalues) @ basis.T
    f = torch.randn((n, 1), generator=generator, dtype=dtype)
    A = torch.ones((1, n), dtype=dtype)
    b = torch.tensor(0.25, dtype=dtype)
    lower = -torch.ones((n, 1), dtype=dtype)
    upper = torch.ones((n, 1), dtype=dtype)
    solution, info = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        lower,
        upper,
        "cpu",
        True,
        {
            "algebra": "torch",
            "settings": {
                "return_info": True,
                "check_linear_residual": True,
                "eps_abs": 1e-8,
                "eps_rel": 1e-8,
                "polishing": False,
            },
        },
        TorchOSQPWorkspace(),
    )
    assert info["status"] == "solved"
    assert info["rho_updates"] > 0
    assert info["primal_residual"] < info["eps_primal"]
    assert info["dual_residual"] < info["eps_dual"]
    assert torch.all(torch.isfinite(solution))
