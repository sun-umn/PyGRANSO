"""Direct KKT solver that follows OSQP's ``qdldl_interface.c`` step by step.

``init_linsys_solver_qdldl`` forms the upper-triangular KKT matrix
(``form_KKT``), orders it with AMD (``permute_KKT``), computes the elimination
tree and the LDL' factors (``LDL_factor``), and ``solve_linsys_qdldl`` applies
``LDLSolve``.  The same sequence is used here so that every stored factor entry
and every solution entry is produced by the same rounded operations as in C.
"""

from __future__ import annotations

import numpy as np
import torch

from pygranso.private.torchKKTForm import form_KKT, permute_KKT
from pygranso.private.torchLinearSolve import (
    LinearSolveDiagnostics,
    TorchLinearSolveError,
    _validate_matrix,
    _validate_rhs,
)
from pygranso.private.torchQDLDL import ldl_solve, qdldl_etree, qdldl_factor


def _scalars(tensor):
    """Entries as scalars whose arithmetic rounds in the tensor's own precision."""

    if tensor.dtype == torch.float64:
        return tensor.detach().cpu().tolist()
    return list(tensor.detach().cpu().numpy().reshape(-1))


def dense_to_csc(matrix, pattern, *, upper_triangular=False):
    """CSC storage of the entries selected by ``pattern`` (rows ascending).

    OSQP receives ``P`` (upper triangle) and ``A`` in CSC form; the stored
    pattern, not the current values, defines the KKT sparsity structure.
    """

    rows, columns = matrix.shape
    mask = pattern.detach().cpu()
    if upper_triangular:
        mask = torch.triu(mask)
    mask_columns = mask.T.tolist()
    if matrix.dtype == torch.float64:
        value_columns = matrix.detach().cpu().T.tolist()
    else:
        value_columns = [list(column) for column in matrix.detach().cpu().numpy().T]
    p, i, x = [0], [], []
    for column in range(columns):
        mask_column = mask_columns[column]
        value_column = value_columns[column]
        for row in range(rows):
            if mask_column[row]:
                i.append(row)
                x.append(value_column[row])
        p.append(len(i))
    return {"m": rows, "n": columns, "p": p, "i": i, "x": x, "nzmax": len(i)}


class QDLDLSolver:
    """Cache the AMD-ordered LDL' factors of the OSQP KKT matrix."""

    solver_name = "QDLDL (Torch translation of qdldl_interface.c, AMD ordering)"

    def __init__(self) -> None:
        self.factorization_count = 0
        self.solve_count = 0
        self._symbolic_key = None
        self._perm = None
        self.clear()

    @property
    def ready(self) -> bool:
        return self._factors is not None

    @property
    def matrix(self) -> torch.Tensor | None:
        return self._matrix

    def clear(self) -> None:
        self._matrix = None
        self._factors = None
        self._info = 0
        self.positive_d_count = None

    def reset(self) -> None:
        self.clear()
        self._symbolic_key = None
        self._perm = None
        self.factorization_count = 0
        self.solve_count = 0

    def is_factorized_for(self, matrix: torch.Tensor) -> bool:
        if not self.ready or self._matrix is None:
            return False
        return (
            self._matrix.shape == matrix.shape
            and self._matrix.device == matrix.device
            and self._matrix.dtype == matrix.dtype
            and torch.equal(self._matrix, matrix)
        )

    def factorize(
        self,
        matrix: torch.Tensor,
        *,
        P: torch.Tensor,
        A: torch.Tensor,
        sigma: float,
        rho_inv_vec: torch.Tensor | None,
        p_pattern: torch.Tensor,
        a_pattern: torch.Tensor,
    ) -> None:
        """``form_KKT`` -> ``permute_KKT`` -> ``LDL_factor`` for ``[P+sigma I, A'; A, -rho_inv]``.

        ``matrix`` is the dense form of the same KKT matrix; it is retained only
        to recognise an unchanged system on the next call.  ``rho_inv_vec`` is
        ``None`` for the polishing system, where C passes the scalar ``sigma``.
        """

        matrix = _validate_matrix(matrix)
        n = P.shape[0]
        m = A.shape[0]
        P_csc = dense_to_csc(P, p_pattern, upper_triangular=True)
        A_csc = dense_to_csc(A, a_pattern)
        if matrix.dtype == torch.float64:
            zero, one = 0.0, 1.0
            sigma_value = float(sigma)
        else:
            zero, one = np.float32(0.0), np.float32(1.0)
            sigma_value = np.float32(sigma)
        param2 = None if rho_inv_vec is None else _scalars(rho_inv_vec)
        KKT = form_KKT(
            P_csc, A_csc, 0, sigma_value, param2, sigma_value, return_maps=False
        )
        symbolic_key = (n, m, tuple(KKT["p"]), tuple(KKT["i"]))
        if self._symbolic_key != symbolic_key or self._perm is None:
            # permute_KKT: AMD ordering of the KKT pattern.
            amd_status, perm, permuted = permute_KKT(KKT)
            if amd_status < 0 or permuted is None:
                raise TorchLinearSolveError("Error forming and permuting KKT matrix.")
            self._symbolic_key = symbolic_key
            self._perm = list(perm)
            self._pinv = None
        else:
            # The ordering depends only on the pattern, as in the C update paths.
            from pygranso.private.torchKKTForm import csc_pinv, csc_symperm

            if self._pinv is None:
                self._pinv = csc_pinv(self._perm)
            permuted = csc_symperm(KKT, self._pinv, None, 1)
        size = n + m
        Kp = [int(v) for v in permuted["p"]]
        Ki = [int(v) for v in permuted["i"]]
        Kx = list(permuted["x"])
        sum_lnz, Lnz, etree = qdldl_etree(size, Kp, Ki)
        if sum_lnz < 0:
            raise TorchLinearSolveError(
                "Error in KKT matrix LDL factorization when computing the "
                "elimination tree."
            )
        positive, Lp, Li, Lx, D, Dinv = qdldl_factor(
            size, Kp, Ki, Kx, Lnz, etree, sum_lnz, zero=zero, one=one
        )
        if positive < 0:
            raise TorchLinearSolveError(
                "Error in KKT matrix LDL factorization when computing the nonzero "
                "elements. There are zeros in the diagonal matrix."
            )
        self._matrix = matrix.detach().clone()
        self._factors = (Lp, Li, Lx, D, Dinv)
        self._kkt = (Kp, Ki, Kx)
        self._symbolic = (Lnz, etree)
        self.positive_d_count = int(positive)
        self.factorization_count += 1

    def solve(self, rhs: torch.Tensor, *, calculate_residual: bool = False):
        if not self.ready or self._matrix is None:
            raise TorchLinearSolveError(
                "The linear solver must be factorized before solve()."
            )
        rhs, vector_rhs = _validate_rhs(rhs, self._matrix)
        Lp, Li, Lx, _, Dinv = self._factors
        columns = []
        for column in rhs.unbind(1):
            values = ldl_solve(_scalars(column), Lp, Li, Lx, Dinv, self._perm)
            columns.append(
                torch.tensor(
                    [float(v) for v in values] if rhs.dtype == torch.float64
                    else np.asarray(values, dtype=np.float32),
                    device=rhs.device,
                    dtype=rhs.dtype,
                )
            )
        solution = (
            torch.stack(columns, dim=1) if columns else torch.empty_like(rhs)
        )
        if not bool(torch.all(torch.isfinite(solution)).item()):
            raise TorchLinearSolveError(
                "QDLDL solve returned a non-finite solution."
            )
        self.solve_count += 1
        residual_norm = None
        relative_residual = None
        if calculate_residual:
            residual = self._matrix @ solution - rhs
            residual_value = torch.linalg.vector_norm(residual)
            denominator = torch.linalg.vector_norm(rhs).clamp_min(1)
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

    # Read-only views used by the C parity tests.
    @property
    def permutation(self):
        return None if self._perm is None else list(self._perm)

    @property
    def factors(self):
        return self._factors

    @property
    def permuted_kkt(self):
        return self._kkt
