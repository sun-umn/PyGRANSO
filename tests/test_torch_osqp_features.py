import torch

from pygranso.private.osqpTorchAdapter import solve_osqp_torch_qp
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchOSQP import (
    _adaptive_rho_update,
    _rho_vector,
    _scale_problem,
    _scaling_for_problem,
    _unscale_state,
)


def test_ruiz_scaling_round_trips_primal_and_dual_state():
    dtype = torch.float64
    P = torch.diag(torch.tensor([1e-4, 1e4], dtype=dtype))
    q = torch.tensor([2.0, -3.0], dtype=dtype)
    A = torch.tensor([[1e3, 0.0], [0.0, 1e-3]], dtype=dtype)
    l = torch.tensor([-1.0, -2.0], dtype=dtype)
    u = torch.tensor([1.0, 2.0], dtype=dtype)
    workspace = TorchOSQPWorkspace()
    scaling = _scaling_for_problem(workspace, P, q, A, {"scaling": 10})
    P_s, q_s, A_s, l_s, u_s = _scale_problem(P, q, A, l, u, scaling)
    assert all(torch.all(torch.isfinite(value)) for value in (P_s, q_s, A_s, l_s, u_s))

    x = torch.tensor([0.25, -0.5], dtype=dtype)
    z = A @ x
    y = torch.tensor([0.3, -0.2], dtype=dtype)
    D, E, cost = scaling["D"], scaling["E"], scaling["cost"]
    x_o, z_o, y_o = _unscale_state(x / D, z * E, y * cost / E, scaling)
    assert torch.allclose(x_o, x)
    assert torch.allclose(z_o, z)
    assert torch.allclose(y_o, y)


def test_adaptive_rho_update_is_deterministic_and_keeps_equality_boost():
    dtype = torch.float64
    rho = torch.tensor(0.1, dtype=dtype)
    equality = torch.tensor([False, True])
    updated, rho_next, rho_vector = _adaptive_rho_update(
        rho,
        equality,
        torch.tensor(10.0, dtype=dtype),
        torch.tensor(0.1, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
        5.0,
    )
    assert updated
    assert rho_next > rho
    assert torch.allclose(rho_vector, _rho_vector(rho_next, equality))
    assert rho_vector[1] == 1000.0 * rho_vector[0]


def test_polishing_handles_duplicate_equality_and_bound_rows():
    dtype = torch.float64
    H = torch.ones((1, 1), dtype=dtype)
    f = torch.tensor([[-2.0]], dtype=dtype)
    A = torch.ones((1, 1), dtype=dtype)
    b = torch.tensor(1.0, dtype=dtype)
    lower = torch.zeros((1, 1), dtype=dtype)
    upper = torch.ones((1, 1), dtype=dtype)
    solution, info = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        lower,
        upper,
        "cpu",
        True,
        {"algebra": "torch", "settings": {"return_info": True}},
        TorchOSQPWorkspace(),
    )
    assert torch.allclose(solution, torch.ones_like(solution), atol=1e-8, rtol=1e-8)
    assert info["polishing_status"] == "accepted"
    assert info["polishing_active_constraints"] == 1
