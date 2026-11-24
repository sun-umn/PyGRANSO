"""Direct KKT solver that follows OSQP's ``qdldl_interface.c`` step by step.

``init_linsys_solver_qdldl`` forms the upper-triangular KKT matrix
(``form_KKT``), orders it with AMD (``permute_KKT``), computes the elimination
tree and the LDL' factors (``LDL_factor``), and ``solve_linsys_qdldl`` applies
``LDLSolve``.  The same sequence is used here so that every stored factor entry
and every solution entry is produced by the same rounded operations as in C.

For CPU tensors the factorization and solves are the scalar loops of
``torchQDLDL``.  For CUDA tensors they are the tensor programs of
``torchQDLDLDevice``, which perform the same operations on the GPU; only the
pattern analysis (KKT pattern, AMD ordering, elimination tree) is done on the
CPU, on integers.
"""

from __future__ import annotations

import numpy as np
import torch

from pygranso.private.torchKKTForm import csc_pinv, csc_symperm, form_KKT, permute_KKT
from pygranso.private.torchLinearSolve import (
    LinearSolveDiagnostics,
    TorchLinearSolveError,
    _validate_rhs,
)
from pygranso.private.torchQDLDL import ldl_solve, qdldl_etree, qdldl_factor
from pygranso.private.torchQDLDLDevice import FactorProgram, SolveProgram

# Test hook: use the tensor programs of torchQDLDLDevice for CPU tensors too.
# CUDA tensors always use them, so their values stay on the GPU.
DEVICE_PROGRAM = False


class _QDLDLNumericError(TorchLinearSolveError):
    """QDLDL_factor returned a negative status."""


def _scalars(tensor):
    """Entries as scalars whose arithmetic rounds in the tensor's own precision."""

    tensor = tensor.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    values = tensor.reshape(-1).tolist()
    if tensor.dtype == torch.float64:
        return values
    return [np.float32(value) for value in values]


def _csc_dict(matrix):
    """``OSQPCscMatrix`` fields of a stored matrix, values as scalars."""

    pointers, rows = matrix.pattern.lists()
    m, n = matrix.shape
    return {
        "m": m, "n": n, "p": pointers, "i": rows,
        "x": _scalars(matrix.values), "nzmax": len(rows),
    }


class QDLDLSolver:
    """Cache the AMD-ordered LDL' factors of the OSQP KKT matrix."""

    solver_name = "QDLDL (Torch translation of qdldl_interface.c, AMD ordering)"

    def __init__(self) -> None:
        self.factorization_count = 0
        self.solve_count = 0
        self._symbolic_key = None
        self._perm = None
        self._device_symbolic = None
        self.clear()

    @property
    def ready(self) -> bool:
        return self._factors is not None

    @property
    def matrix(self):
        """The ``KKTSystem`` of the last factorization attempt."""
        return self._matrix

    def clear(self) -> None:
        self._matrix = None
        self._factors = None
        self._info = 0
        self.positive_d_count = None

    def invalidate(self) -> None:
        """Retain the attempted KKT but require a rebuild before using it."""
        self._factors = None
        self._info = 1

    def reset(self) -> None:
        self.clear()
        self._symbolic_key = None
        self._perm = None
        self._device_symbolic = None
        self.factorization_count = 0
        self.solve_count = 0

    def is_factorized_for(self, matrix) -> bool:
        if not self.ready or self._matrix is None:
            return False
        return self._matrix.equal(matrix)

    def factorize(self, matrix) -> None:
        """``form_KKT`` -> ``permute_KKT`` -> ``LDL_factor`` for a ``KKTSystem``.

        The system is built from the stored CSC entries of ``P`` and ``A``; it
        is retained to recognise an unchanged system on the next call.
        """

        P, A = matrix.P, matrix.A
        sigma, rho_inv_vec = matrix.sigma, matrix.rho_inv_vec
        for values in (P.values, A.values, rho_inv_vec):
            if values is not None and not bool(torch.all(torch.isfinite(values)).item()):
                raise ValueError("The linear-system matrix contains NaN or Inf.")
        if matrix.device.type != "cpu" or DEVICE_PROGRAM:
            self._factorize_on_device(matrix)
            return
        n = P.shape[0]
        m = A.shape[0]
        P_csc = _csc_dict(P)
        A_csc = _csc_dict(A)
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
            self._matrix = matrix
            self._kkt = (Kp, Ki, Kx)
            self._symbolic = (Lnz, etree)
            self.positive_d_count = None
            self.invalidate()
            raise _QDLDLNumericError(
                "Error in KKT matrix LDL factorization when computing the nonzero "
                "elements. There are zeros in the diagonal matrix."
            )
        self._matrix = matrix
        self._factors = (Lp, Li, Lx, D, Dinv)
        self._kkt = (Kp, Ki, Kx)
        self._symbolic = (Lnz, etree)
        self.positive_d_count = int(positive)
        self.factorization_count += 1

    def _device_symbolic_for(self, matrix):
        """Pattern-only setup: KKT pattern and maps, AMD ordering, etree, programs."""

        P, A = matrix.P, matrix.A
        cached = self._device_symbolic
        if (
            cached is not None
            and cached["device"] == matrix.device
            and cached["P"].same(P.pattern)
            and cached["A"].same(A.pattern)
        ):
            return cached
        n, m = P.shape[0], A.shape[0]
        device = matrix.device

        def pattern_only(stored):
            pointers, rows = stored.pattern.lists()
            return {
                "m": stored.shape[0], "n": stored.shape[1], "p": pointers,
                "i": rows, "x": [0.0] * len(rows), "nzmax": len(rows),
            }

        KKT, PtoKKT, AtoKKT, rhotoKKT = form_KKT(
            pattern_only(P), pattern_only(A), 0, 0.0, None, 0.0, return_maps=True
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
        if self._pinv is None:
            self._pinv = csc_pinv(self._perm)
        KtoPKPt = [0] * len(KKT["i"])
        permuted = csc_symperm(KKT, self._pinv, KtoPKPt, 0)
        size = n + m
        Kp = [int(v) for v in permuted["p"]]
        Ki = [int(v) for v in permuted["i"]]
        sum_lnz, Lnz, etree = qdldl_etree(size, Kp, Ki)
        if sum_lnz < 0:
            raise TorchLinearSolveError(
                "Error in KKT matrix LDL factorization when computing the "
                "elimination tree."
            )

        def index(values):
            return torch.tensor(values, dtype=torch.int64, device=device)

        factor = FactorProgram(size, Kp, Ki, Lnz, etree, sum_lnz, device)
        cached = {
            "device": device, "P": P.pattern, "A": A.pattern,
            "nnz": len(KKT["i"]),
            "PtoKKT": index(PtoKKT), "AtoKKT": index(AtoKKT),
            "rhotoKKT": index(rhotoKKT),
            # _kkt_shifts_param1: the diagonal is the last entry of each column.
            "diagonal": index([KKT["p"][i + 1] - 1 for i in range(n)]),
            "KtoPKPt": index(KtoPKPt),
            "Kp": Kp, "Ki": Ki, "Lnz": Lnz, "etree": etree,
            "factor": factor,
            "solve": SolveProgram(size, factor.Lp, factor.Li, self._perm, device),
        }
        self._device_symbolic = cached
        return cached

    def _factorize_on_device(self, matrix) -> None:
        """``factorize`` with every value kept on the device of ``matrix``."""

        P, A = matrix.P, matrix.A
        symbolic = self._device_symbolic_for(matrix)
        # form_KKT: P, structural zeros on the diagonal, A', then the shifts.
        values = P.values.new_zeros(symbolic["nnz"])
        values[symbolic["PtoKKT"]] = P.values
        values[symbolic["AtoKKT"]] = A.values
        diagonal = symbolic["diagonal"]
        values[diagonal] = values[diagonal] + matrix.sigma
        rho = symbolic["rhotoKKT"]
        values[rho] = values[rho] - (
            matrix.sigma if matrix.rho_inv_vec is None else matrix.rho_inv_vec
        )
        # csc_symperm
        Kx = torch.empty_like(values)
        Kx[symbolic["KtoPKPt"]] = values
        Lx, D, Dinv = symbolic["factor"].run(Kx)
        factor = symbolic["factor"]
        self._matrix = matrix
        self._kkt = (symbolic["Kp"], symbolic["Ki"], Kx)
        self._symbolic = (symbolic["Lnz"], symbolic["etree"])
        if bool((D == 0).any().item()):
            # QDLDL_factor returns -1 at the first zero entry of D.
            self.positive_d_count = None
            self.invalidate()
            raise _QDLDLNumericError(
                "Error in KKT matrix LDL factorization when computing the nonzero "
                "elements. There are zeros in the diagonal matrix."
            )
        self._factors = (factor.Lp, factor.Li, Lx, D, Dinv)
        self._program = (symbolic["solve"], symbolic["solve"].bind(Lx))
        self.positive_d_count = int((D > 0).sum().item())
        self.factorization_count += 1

    def solve(self, rhs: torch.Tensor, *, calculate_residual: bool = False):
        if not self.ready or self._matrix is None:
            raise TorchLinearSolveError(
                "The linear solver must be factorized before solve()."
            )
        rhs, vector_rhs = _validate_rhs(rhs, self._matrix)
        Lp, Li, Lx, _, Dinv = self._factors
        columns = []
        on_device = torch.is_tensor(Lx)
        for column in rhs.unbind(1):
            if on_device:
                program, bound = self._program
                columns.append(program.run(bound, Dinv, column))
                continue
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
            residual = torch.stack(
                [self._matrix.matvec(column) for column in solution.unbind(1)], dim=1
            ) - rhs
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
