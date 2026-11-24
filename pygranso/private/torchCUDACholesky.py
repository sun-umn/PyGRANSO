"""CUDA direct solve by eliminating the negative diagonal of the KKT system."""

import torch

from pygranso.private.torchLinearSolve import LinearSolveDiagnostics
from pygranso.private.torchQDLDLSolve import QDLDLSolver, _QDLDLNumericError


class CUDACholeskySolver(QDLDLSolver):
    """Share workspace cache semantics; numeric factors and every RHS stay on CUDA."""

    solver_name = "CUDA reduced KKT Cholesky"

    def clear(self):
        super().clear()
        self._shift_basis = None

    def factorize(self, matrix):
        self._shift_basis = None
        A = matrix.A.to_dense()
        inv = matrix.rho_inv_vec
        if inv is None:  # polishing uses -sigma I for the lower block
            inv = A.new_full((A.shape[0],), matrix.sigma)
        weights = inv.reciprocal().unsqueeze(1)
        reduced = matrix.P.to_dense()
        reduced.diagonal().add_(matrix.sigma)
        reduced.addmm_(A.T, weights * A)
        lower, info = torch.linalg.cholesky_ex(reduced, check_errors=False)
        self._matrix = matrix
        if int(info) != 0:
            self.invalidate()
            raise _QDLDLNumericError(
                "KKT inertia check failed during reduced Cholesky factorization."
            )
        self._factors = (A, weights, lower)
        # The eliminated block is negative definite and its Schur complement is SPD.
        self.positive_d_count = reduced.shape[0]
        self._info = 0
        self.factorization_count += 1

    def prepare_rho_shifts(self, rho, equality, loose):
        """Diagonalize the rho-dependent part once for off-check updates.

        If L L' = H(r0), then H(r) = L [I + (r-r0) B] L', with
        B = L^-1 A' W A L^-T. Loose rows have constant weight, so W is
        zero there. The new inverse diagonal can be selected on device.
        """
        if self._shift_basis is not None:
            return
        A, _, lower = self._factors
        rows = torch.where(equality, 1000.0, 1.0).to(A.dtype)
        rows = torch.where(loose, 0.0, rows)
        whitened = torch.linalg.solve_triangular(lower, A.T, upper=False)
        _, vectors = torch.linalg.eigh((whitened * rows) @ whitened.T)
        basis = torch.linalg.solve_triangular(lower.T, vectors, upper=True)
        # Evaluate both coefficients of diag(T' H(r) T) directly. The shifted
        # form 1 + (r-r0)*lambda can lose all digits after a large rho decrease.
        projected_a = A @ basis
        squares = projected_a.square()
        slope = rows @ squares
        base = self.matrix.P.to_dense()
        base.diagonal().add_(self.matrix.sigma)
        intercept = (basis * (base @ basis)).sum(0) + (
            loose.to(A.dtype) * 1e-6
        ) @ squares
        self._shift_basis = (intercept, slope, basis)
        self._inverse_shift = (intercept + rho * slope).reciprocal()
        self._valid_shifts = rho.new_ones((), dtype=torch.bool)

    def set_shift(self, matrix, rho, accepted):
        intercept, slope, _ = self._shift_basis
        diagonal = intercept + rho * slope
        self._valid_shifts = self._valid_shifts & (diagonal > 0).all()
        self._inverse_shift = torch.where(
            accepted, diagonal.reciprocal(), self._inverse_shift
        )
        A, _, lower = self._factors
        self._factors = A, matrix.rho_inv_vec.reciprocal().unsqueeze(1), lower
        self._matrix = matrix

    def check_shifts(self):
        if not bool(self._valid_shifts):
            raise _QDLDLNumericError("Indefinite reduced KKT rho update")

    def solve(self, rhs, *, calculate_residual=False):
        if not self.ready:
            raise RuntimeError("The CUDA KKT system must be factorized before solve().")
        A, weights, lower = self._factors
        vector = rhs.ndim == 1
        columns = rhs.unsqueeze(1) if vector else rhs
        n = lower.shape[0]
        b, c = columns[:n], columns[n:]
        reduced_rhs = b + A.T @ (weights * c)
        if self._shift_basis is None:
            forward = torch.linalg.solve_triangular(lower, reduced_rhs, upper=False)
            x = torch.linalg.solve_triangular(lower.T, forward, upper=True)
        else:
            basis = self._shift_basis[2]
            x = basis @ (self._inverse_shift[:, None] * (basis.T @ reduced_rhs))
        solution = torch.cat((x, weights * (A @ x - c)))
        self.solve_count += 1
        norm = relative = None
        if calculate_residual:
            residual = (
                torch.stack([self.matrix.matvec(v) for v in solution.T], dim=1)
                - columns
            )
            norm = torch.linalg.vector_norm(residual)
            relative = norm / torch.linalg.vector_norm(columns).clamp_min(1)
        diagnostics = LinearSolveDiagnostics(
            self.solver_name,
            0,
            self.factorization_count,
            self.solve_count,
            norm,
            relative,
        )
        return (solution[:, 0] if vector else solution), diagnostics.as_dict()
