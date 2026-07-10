import torch

from pygranso.private.torchOSQP import (
    admm_vector_update,
    build_kkt_matrix,
    build_kkt_rhs,
    recover_z_tilde,
)


def test_kkt_blocks_rhs_and_recovery():
    P = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    A = torch.tensor([[1.0, -1.0], [0.0, 1.0]], dtype=torch.float64)
    rho = torch.tensor([0.1, 100.0], dtype=torch.float64)
    sigma = 1e-6
    K = build_kkt_matrix(P, A, sigma, rho)
    assert K.shape == (4, 4)
    assert torch.allclose(K, K.T)
    assert torch.allclose(K[:2, :2], P + sigma * torch.eye(2, dtype=P.dtype))
    assert torch.allclose(K[:2, 2:], A.T)
    assert torch.allclose(K[2:, :2], A)
    assert torch.allclose(K[2:, 2:], -torch.diag(rho.reciprocal()))

    x = torch.tensor([0.3, -0.2], dtype=P.dtype)
    z = torch.tensor([0.1, 0.4], dtype=P.dtype)
    y = torch.tensor([0.2, -0.5], dtype=P.dtype)
    q = torch.tensor([-1.0, 2.0], dtype=P.dtype)
    rhs = build_kkt_rhs(x, z, y, q, sigma, rho)
    solution = torch.linalg.solve(K, rhs)
    z_tilde = recover_z_tilde(z, solution[2:], y, rho)
    assert torch.allclose(z_tilde, A @ solution[:2])


def test_shared_admm_vector_update_matches_equations():
    x_tilde = torch.tensor([1.0])
    x = torch.tensor([0.0])
    z_tilde = torch.tensor([2.0])
    z = torch.tensor([0.5])
    y = torch.tensor([0.2])
    rho = torch.tensor([0.1])
    lower = torch.tensor([0.0])
    upper = torch.tensor([1.0])
    x_next, z_next, y_next = admm_vector_update(
        x_tilde, x, z_tilde, z, y, rho, lower, upper, 1.6
    )
    assert torch.allclose(x_next, torch.tensor([1.6]))
    assert torch.allclose(z_next, torch.tensor([1.0]))
    assert torch.allclose(y_next, torch.tensor([0.39]))
