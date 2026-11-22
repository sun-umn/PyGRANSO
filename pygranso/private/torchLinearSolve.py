"""Common validation and diagnostics for the OSQP linear solvers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class TorchLinearSolveError(RuntimeError):
    """Raised when PyTorch cannot factor or solve a validated linear system."""


@dataclass(frozen=True)
class LinearSolveDiagnostics:
    solver: str
    factorization_info: int
    factorization_count: int
    solve_count: int
    linear_residual_norm: float | None = None
    relative_linear_residual: float | None = None

    def as_dict(self) -> dict:
        return {
            "solver": self.solver,
            "factorization_info": self.factorization_info,
            "factorization_count": self.factorization_count,
            "solve_count": self.solve_count,
            "linear_residual_norm": self.linear_residual_norm,
            "relative_linear_residual": self.relative_linear_residual,
        }


def _validate_matrix(matrix: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(matrix):
        raise TypeError("The linear-system matrix must be a Torch tensor.")
    matrix = matrix.detach()
    if matrix.layout != torch.strided:
        raise ValueError("The linear solver requires a strided matrix.")
    if matrix.ndim != 2:
        raise ValueError("The linear-system matrix must be two-dimensional.")
    rows, columns = matrix.shape
    if rows != columns:
        raise ValueError("The linear-system matrix must be square.")
    if matrix.dtype not in {torch.float32, torch.float64}:
        raise ValueError("The linear solver supports float32 and float64 only.")
    if not bool(torch.all(torch.isfinite(matrix)).item()):
        raise ValueError("The linear-system matrix contains NaN or Inf.")
    return matrix


def _validate_rhs(
    rhs: torch.Tensor,
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    if not torch.is_tensor(rhs):
        raise TypeError("The linear-system right-hand side must be a Torch tensor.")
    rhs = rhs.detach()
    if rhs.ndim not in {1, 2}:
        raise ValueError("The right-hand side must be a vector or matrix.")
    if rhs.shape[0] != matrix.shape[0]:
        raise ValueError("The right-hand side is incompatible with the matrix.")
    if rhs.device != matrix.device:
        raise ValueError("The matrix and right-hand side must use the same device.")
    if rhs.dtype != matrix.dtype:
        raise ValueError("The matrix and right-hand side must use the same dtype.")
    if not bool(torch.all(torch.isfinite(rhs)).item()):
        raise ValueError("The linear-system right-hand side contains NaN or Inf.")
    vector_rhs = rhs.ndim == 1
    if vector_rhs:
        rhs = rhs.reshape(-1, 1)
    return rhs, vector_rhs
