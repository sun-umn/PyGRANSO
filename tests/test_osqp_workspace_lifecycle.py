import torch

from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchOSQP import _prepare_workspace


def _seed_workspace(workspace):
    workspace.state = {
        "x": torch.ones(2),
        "z": torch.ones(2),
        "y": torch.ones(2),
    }
    workspace.rho_bar = 0.25
    workspace.rho_setting = 0.1


def test_compatible_matrix_value_update_preserves_warm_state():
    workspace = TorchOSQPWorkspace()
    P = torch.eye(2, dtype=torch.float64)
    A = torch.eye(2, dtype=torch.float64)
    _prepare_workspace(workspace, P, A, ("bounds", 2))
    _seed_workspace(workspace)
    _prepare_workspace(workspace, 2.0 * P, 3.0 * A, ("bounds", 2))
    assert workspace.state is not None
    assert workspace.rho_bar == 0.25


def test_structure_change_invalidates_complete_torch_state():
    workspace = TorchOSQPWorkspace()
    P = torch.eye(2, dtype=torch.float64)
    A = torch.eye(2, dtype=torch.float64)
    _prepare_workspace(workspace, P, A, ("bounds", 2))
    _seed_workspace(workspace)
    changed = A.clone()
    changed[0, 1] = 1.0
    _prepare_workspace(workspace, P, changed, ("bounds", 2))
    assert workspace.state is None
    assert workspace.rho_bar is None
    assert not workspace.linear_solver.ready


def test_constraint_order_signature_change_invalidates_complete_torch_state():
    workspace = TorchOSQPWorkspace()
    P = torch.eye(2, dtype=torch.float64)
    A = torch.ones((2, 2), dtype=torch.float64)
    _prepare_workspace(workspace, P, A, ("c1", "c2"))
    _seed_workspace(workspace)
    _prepare_workspace(workspace, P, A, ("c2", "c1"))
    assert workspace.state is None
    assert workspace.constraint_order_signature == ("c2", "c1")


def test_dimension_and_dtype_changes_invalidate_complete_torch_state():
    for changed_p, changed_a in (
        (torch.eye(3, dtype=torch.float64), torch.eye(3, dtype=torch.float64)),
        (torch.eye(2, dtype=torch.float32), torch.eye(2, dtype=torch.float32)),
    ):
        workspace = TorchOSQPWorkspace()
        _prepare_workspace(
            workspace,
            torch.eye(2, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64),
            ("bounds", 2),
        )
        _seed_workspace(workspace)
        _prepare_workspace(workspace, changed_p, changed_a, ("changed",))
        assert workspace.state is None
        assert workspace.rho_bar is None


def test_a_value_update_preserves_state_but_forces_new_factors():
    from pygranso.private.osqpTorchAdapter import solve_osqp_torch_qp

    dtype = torch.float64
    workspace = TorchOSQPWorkspace()
    H = torch.eye(2, dtype=dtype)
    f = torch.zeros((2, 1), dtype=dtype)
    lower = -torch.ones((2, 1), dtype=dtype)
    upper = torch.ones((2, 1), dtype=dtype)
    settings = {"return_info": True, "polishing": False}
    _, first = solve_osqp_torch_qp(
        H,
        f,
        torch.tensor([[1.0, 1.0]], dtype=dtype),
        torch.tensor(0.0, dtype=dtype),
        lower,
        upper,
        "cpu",
        True,
        {"algebra": "torch", "settings": settings},
        workspace,
    )
    prior_state = {key: value.clone() for key, value in workspace.state.items()}
    _, second = solve_osqp_torch_qp(
        H,
        f,
        torch.tensor([[2.0, 1.0]], dtype=dtype),
        torch.tensor(0.0, dtype=dtype),
        lower,
        upper,
        "cpu",
        True,
        {"algebra": "torch", "settings": settings},
        workspace,
    )
    assert first["factorizations_this_solve"] >= 1
    assert second["factorizations_this_solve"] >= 1
    assert prior_state.keys() == workspace.state.keys()
