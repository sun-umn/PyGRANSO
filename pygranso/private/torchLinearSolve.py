"""Validated PyTorch-native dense linear-system factorization and solves."""

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


class DenseLUSolver:
    """Cache a dense LU factorization and solve repeated right-hand sides."""

    solver_name = "torch.linalg.lu_factor_ex/lu_solve"

    def __init__(self) -> None:
        self._matrix: torch.Tensor | None = None
        self._lu: torch.Tensor | None = None
        self._pivots: torch.Tensor | None = None
        self._info = 0
        self.factorization_count = 0
        self.solve_count = 0

    @property
    def ready(self) -> bool:
        return self._lu is not None and self._pivots is not None

    @property
    def matrix(self) -> torch.Tensor | None:
        return self._matrix

    def clear(self) -> None:
        self._matrix = None
        self._lu = None
        self._pivots = None
        self._info = 0

    def is_factorized_for(self, matrix: torch.Tensor) -> bool:
        if not self.ready or self._matrix is None:
            return False
        return (
            self._matrix.shape == matrix.shape
            and self._matrix.device == matrix.device
            and self._matrix.dtype == matrix.dtype
            and torch.equal(self._matrix, matrix)
        )

    def factorize(self, matrix: torch.Tensor) -> None:
        matrix = _validate_matrix(matrix)
        try:
            lu, pivots, info = torch.linalg.lu_factor_ex(
                matrix,
                check_errors=False,
            )
        except (NotImplementedError, RuntimeError) as exc:
            raise TorchLinearSolveError(
                f"PyTorch LU factorization is unavailable or failed on "
                f"{matrix.device}: {exc}"
            ) from exc

        info_value = int(info.detach().cpu().item())
        if info_value != 0:
            raise TorchLinearSolveError(
                "PyTorch LU factorization reported a singular or invalid "
                f"matrix (info={info_value})."
            )
        if not bool(torch.all(torch.isfinite(lu)).item()):
            raise TorchLinearSolveError(
                "PyTorch LU factorization returned non-finite factors."
            )

        self._matrix = matrix.detach().clone()
        self._lu = lu
        self._pivots = pivots
        self._info = info_value
        self.factorization_count += 1

    def refactorize(self, matrix: torch.Tensor) -> None:
        self.factorize(matrix)

    def factorize_if_needed(self, matrix: torch.Tensor) -> bool:
        if self.is_factorized_for(matrix):
            return False
        self.factorize(matrix)
        return True

    def solve(
        self,
        rhs: torch.Tensor,
        *,
        calculate_residual: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        if not self.ready or self._matrix is None:
            raise TorchLinearSolveError(
                "The linear solver must be factorized before solve()."
            )
        rhs, vector_rhs = _validate_rhs(rhs, self._matrix)
        try:
            solution = torch.linalg.lu_solve(self._lu, self._pivots, rhs)
        except (NotImplementedError, RuntimeError) as exc:
            raise TorchLinearSolveError(
                f"PyTorch LU solve failed on {rhs.device}: {exc}"
            ) from exc
        if not bool(torch.all(torch.isfinite(solution)).item()):
            raise TorchLinearSolveError(
                "PyTorch LU solve returned a non-finite solution."
            )

        self.solve_count += 1
        residual_norm = None
        relative_residual = None
        if calculate_residual:
            residual = self._matrix @ solution - rhs
            residual_value = torch.linalg.vector_norm(residual)
            rhs_value = torch.linalg.vector_norm(rhs)
            denominator = torch.maximum(
                rhs_value,
                torch.ones((), device=rhs.device, dtype=rhs.dtype),
            )
            residual_norm = float(residual_value.item())
            relative_residual = float((residual_value / denominator).item())

        diagnostics = LinearSolveDiagnostics(
            solver=self.solver_name,
            factorization_info=self._info,
            factorization_count=self.factorization_count,
            solve_count=self.solve_count,
            linear_residual_norm=residual_norm,
            relative_linear_residual=relative_residual,
        ).as_dict()
        if vector_rhs:
            solution = solution.reshape(-1)
        return solution, diagnostics


def _validate_matrix(matrix: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(matrix):
        raise TypeError("The linear-system matrix must be a Torch tensor.")
    matrix = matrix.detach()
    if matrix.layout != torch.strided:
        raise ValueError("The dense LU solver requires a strided matrix.")
    if matrix.ndim != 2:
        raise ValueError("The linear-system matrix must be two-dimensional.")
    rows, columns = matrix.shape
    if rows != columns:
        raise ValueError("The linear-system matrix must be square.")
    if matrix.dtype not in {torch.float32, torch.float64}:
        raise ValueError("The dense LU solver supports float32 and float64 only.")
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
