"""Experimental cuDSS-backed solve adapter; no reusable factors are exposed."""

import torch

from pygranso.private.torchLinearSolve import (
    LinearSolveDiagnostics,
    TorchLinearSolveError,
    _validate_matrix,
    _validate_rhs,
)


class SparseDirectSolver:
    """Retain CSR data, but perform analysis/factorization on every spsolve."""

    solver_name = "torch.sparse.spsolve (cuDSS; one-shot)"

    def __init__(self):
        self.reset()

    def clear(self):
        self._matrix = None
        self._csr = None

    def reset(self):
        self.clear()
        self.factorization_count = 0
        self.solve_count = 0

    @property
    def ready(self):
        return self._csr is not None

    @property
    def matrix(self):
        return self._matrix

    def is_factorized_for(self, matrix):
        # Compatibility with the workspace interface: only matrix data is cached.
        return (
            self.ready
            and self._matrix.shape == matrix.shape
            and self._matrix.device == matrix.device
            and self._matrix.dtype == matrix.dtype
            and torch.equal(self._matrix, matrix)
        )

    def _solve_vector(self, csr, rhs):
        try:
            solution = torch.sparse.spsolve(csr, rhs.contiguous())
        except (NotImplementedError, RuntimeError) as exc:
            raise TorchLinearSolveError(
                "Sparse direct solve requires a working CUDA/cuDSS PyTorch build; "
                f"spsolve failed: {exc}"
            ) from exc
        if not bool(torch.isfinite(solution).all().item()):
            raise TorchLinearSolveError(
                "Sparse direct solve returned non-finite values."
            )
        self.factorization_count += 1
        return solution

    def factorize(self, matrix):
        matrix = _validate_matrix(matrix)
        if matrix.device.type != "cuda" or torch.version.hip is not None:
            raise TorchLinearSolveError("use_sparse requires NVIDIA CUDA and cuDSS.")
        # OSQP reports setup/update failures before entering the ADMM loop.
        # spsolve has no factor-only API: a zero-RHS solve validates this stage.
        candidate = matrix.detach().clone()
        csr = candidate.to_sparse_csr()
        self._solve_vector(csr, candidate.new_zeros(candidate.shape[0]))
        self._matrix, self._csr = candidate, csr

    def refactorize(self, matrix):
        self.factorize(matrix)

    def factorize_if_needed(self, matrix):
        if self.is_factorized_for(matrix):
            return False
        self.factorize(matrix)
        return True

    def solve(self, rhs, *, calculate_residual=False):
        if not self.ready:
            raise TorchLinearSolveError(
                "The sparse solver must be initialized before solve()."
            )
        rhs, vector_rhs = _validate_rhs(rhs, self._matrix)
        solution = (
            torch.stack(
                [self._solve_vector(self._csr, column) for column in rhs.unbind(1)],
                dim=1,
            )
            if rhs.shape[1]
            else torch.empty_like(rhs)
        )
        self.solve_count += 1
        residual_norm = relative_residual = None
        if calculate_residual:
            residual = torch.linalg.vector_norm(
                torch.sparse.mm(self._csr, solution) - rhs
            )
            denominator = torch.linalg.vector_norm(rhs).clamp_min(1)
            residual_norm = float(residual.item())
            relative_residual = float((residual / denominator).item())
        info = LinearSolveDiagnostics(
            self.solver_name,
            0,
            self.factorization_count,
            self.solve_count,
            residual_norm,
            relative_residual,
        ).as_dict()
        return (solution.reshape(-1) if vector_rhs else solution), info
