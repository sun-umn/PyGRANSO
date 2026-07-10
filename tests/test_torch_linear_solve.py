import pytest
import torch

from pygranso.private.torchLinearSolve import DenseLUSolver, TorchLinearSolveError


def test_known_system_and_multiple_rhs(floating_dtype, available_device):
    matrix = torch.tensor(
        [[4.0, 1.0], [2.0, 3.0]],
        device=available_device,
        dtype=floating_dtype,
    )
    rhs = torch.tensor(
        [[1.0, 2.0], [0.0, 1.0]],
        device=available_device,
        dtype=floating_dtype,
    )
    solver = DenseLUSolver()
    solver.factorize(matrix)
    solution, info = solver.solve(rhs, calculate_residual=True)
    tolerance = 1e-5 if floating_dtype == torch.float32 else 1e-12
    assert torch.allclose(matrix @ solution, rhs, rtol=tolerance, atol=tolerance)
    assert info["relative_linear_residual"] <= tolerance


def test_factorization_is_reused():
    matrix = torch.eye(3, dtype=torch.float64)
    solver = DenseLUSolver()
    assert solver.factorize_if_needed(matrix)
    assert not solver.factorize_if_needed(matrix.clone())
    solver.solve(torch.ones(3, dtype=torch.float64))
    assert solver.factorization_count == 1
    assert solver.solve_count == 1


def test_vector_rhs_is_normalized_internally_and_shape_is_restored():
    matrix = torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    solver = DenseLUSolver()
    solver.factorize(matrix)
    vector_solution, _ = solver.solve(torch.ones(2, dtype=torch.float64))
    column_solution, _ = solver.solve(torch.ones((2, 1), dtype=torch.float64))
    assert vector_solution.shape == (2,)
    assert column_solution.shape == (2, 1)
    assert torch.allclose(vector_solution, column_solution.reshape(-1))


@pytest.mark.parametrize(
    "matrix,rhs,error",
    [
        (torch.ones(2), torch.ones(2), "two-dimensional"),
        (torch.ones((2, 3)), torch.ones(2), "square"),
        (torch.eye(2), torch.ones(3), "incompatible"),
    ],
)
def test_shape_validation(matrix, rhs, error):
    solver = DenseLUSolver()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        with pytest.raises(ValueError, match=error):
            solver.factorize(matrix)
        return
    solver.factorize(matrix)
    with pytest.raises(ValueError, match=error):
        solver.solve(rhs)


def test_singular_matrix_reports_factorization_failure():
    solver = DenseLUSolver()
    matrix = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(TorchLinearSolveError, match="singular|invalid"):
        solver.factorize(matrix)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_inputs_are_rejected_before_lu(bad):
    solver = DenseLUSolver()
    matrix = torch.eye(2)
    matrix[0, 0] = bad
    with pytest.raises(ValueError, match="NaN or Inf"):
        solver.factorize(matrix)

    solver.factorize(torch.eye(2))
    rhs = torch.ones(2)
    rhs[0] = bad
    with pytest.raises(ValueError, match="NaN or Inf"):
        solver.solve(rhs)
