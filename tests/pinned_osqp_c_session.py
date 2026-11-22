"""Test-only persistent session on the pinned OSQP C library.

``pinned_osqp_c_oracle.solve_dense_qp`` performs one setup and one solve.  The
session here keeps the C solver alive so that repeated solves, vector updates
and explicit warm starts can be compared against the Torch translation.
"""

from __future__ import annotations

import ctypes as ct

import numpy as np
from pinned_osqp_c_oracle import (
    OSQPFloat,
    OSQPInt,
    OSQPSettings,
    OSQPSolver,
    _as_float,
    _pointer,
    dense_to_csc,
    load_pinned_library,
)

OSQP_INFTY = 1e30


def _bind(library):
    library.osqp_update_data_vec.restype = OSQPInt
    library.osqp_update_data_vec.argtypes = [
        ct.POINTER(OSQPSolver),
        ct.POINTER(OSQPFloat),
        ct.POINTER(OSQPFloat),
        ct.POINTER(OSQPFloat),
    ]
    library.osqp_update_data_mat.restype = OSQPInt
    library.osqp_update_data_mat.argtypes = [
        ct.POINTER(OSQPSolver),
        ct.POINTER(OSQPFloat), ct.POINTER(OSQPInt), OSQPInt,
        ct.POINTER(OSQPFloat), ct.POINTER(OSQPInt), OSQPInt,
    ]
    library.osqp_warm_start.restype = OSQPInt
    library.osqp_warm_start.argtypes = [
        ct.POINTER(OSQPSolver),
        ct.POINTER(OSQPFloat),
        ct.POINTER(OSQPFloat),
    ]
    return library


class PinnedOSQPSession:
    """One ``osqp_setup`` followed by any number of solves and updates."""

    def __init__(self, P, q, A, l, u, **settings_overrides):
        self.library = _bind(load_pinned_library())
        self.n = int(np.asarray(q).size)
        self.m = int(np.asarray(l).size)
        self._q = _as_float(q).reshape(-1).copy()
        self._l = _as_float(l).reshape(-1).copy()
        self._u = _as_float(u).reshape(-1).copy()
        self._P = dense_to_csc(np.asarray(P, dtype=np.float64), upper_triangular=True)
        self._A = dense_to_csc(np.asarray(A, dtype=np.float64).reshape(self.m, self.n))
        library = self.library
        self._P_csc = library.OSQPCscMatrix_new(
            self.n, self.n, self._P[0].size,
            _pointer(self._P[0], OSQPFloat),
            _pointer(self._P[1], OSQPInt),
            _pointer(self._P[2], OSQPInt),
        )
        self._A_csc = library.OSQPCscMatrix_new(
            self.m, self.n, self._A[0].size,
            _pointer(self._A[0], OSQPFloat),
            _pointer(self._A[1], OSQPInt),
            _pointer(self._A[2], OSQPInt),
        )
        self._settings = library.OSQPSettings_new()
        self._settings.contents.verbose = 0
        known = dict(OSQPSettings._fields_)
        for name, value in settings_overrides.items():
            if name not in known:
                raise ValueError(f"Unknown pinned OSQP setting {name!r}.")
            setattr(self._settings.contents, name, value)
        self._solver = ct.POINTER(OSQPSolver)()
        self.setup_code = int(
            library.osqp_setup(
                ct.byref(self._solver),
                self._P_csc,
                _pointer(self._q, OSQPFloat),
                self._A_csc,
                _pointer(self._l, OSQPFloat),
                _pointer(self._u, OSQPFloat),
                self.m,
                self.n,
                self._settings,
            )
        )

    def update(self, q=None, l=None, u=None):
        """``osqp_update_data_vec``; returns the C exit flag."""

        arrays = [None if v is None else _as_float(v).reshape(-1).copy() for v in (q, l, u)]
        pointers = [None if a is None else _pointer(a, OSQPFloat) for a in arrays]
        return int(self.library.osqp_update_data_vec(self._solver, *pointers))

    @staticmethod
    def _values_in_stored_order(dense, stored):
        _, rows, pointers = stored
        dense = np.asarray(dense, dtype=np.float64)
        values = [
            dense[rows[k], column]
            for column in range(len(pointers) - 1)
            for k in range(pointers[column], pointers[column + 1])
        ]
        return _as_float(values)

    def update_matrices(self, P=None, A=None):
        """``osqp_update_data_mat`` with all stored values of P and/or A replaced.

        The new dense matrices must keep the sparsity pattern given at setup.
        """

        px = None if P is None else self._values_in_stored_order(P, self._P)
        ax = None if A is None else self._values_in_stored_order(
            np.asarray(A, dtype=np.float64).reshape(self.m, self.n), self._A
        )
        return int(
            self.library.osqp_update_data_mat(
                self._solver,
                None if px is None else _pointer(px, OSQPFloat), None,
                0 if px is None else px.size,
                None if ax is None else _pointer(ax, OSQPFloat), None,
                0 if ax is None else ax.size,
            )
        )

    def warm_start(self, x=None, y=None):
        """``osqp_warm_start``; returns the C exit flag."""

        arrays = [None if v is None else _as_float(v).reshape(-1).copy() for v in (x, y)]
        pointers = [None if a is None else _pointer(a, OSQPFloat) for a in arrays]
        return int(self.library.osqp_warm_start(self._solver, *pointers))

    def solve(self):
        solve_code = int(self.library.osqp_solve(self._solver))
        info = self._solver.contents.info.contents
        solution = self._solver.contents.solution.contents
        settings = self._solver.contents.settings.contents

        def vector(pointer, size):
            return np.ctypeslib.as_array(pointer, shape=(size,)).copy()

        return {
            "solve_code": solve_code,
            "status": info.status.split(b"\0", 1)[0].decode(),
            "status_val": int(info.status_val),
            "status_polish": int(info.status_polish),
            "iter": int(info.iter),
            "rho_updates": int(info.rho_updates),
            "rho_estimate": float(info.rho_estimate),
            "rho": float(settings.rho),
            "objective": float(info.obj_val),
            "dual_objective": float(info.dual_obj_val),
            "duality_gap": float(info.duality_gap),
            "primal_residual": float(info.prim_res),
            "dual_residual": float(info.dual_res),
            "rel_kkt_error": float(info.rel_kkt_error),
            "x": vector(solution.x, self.n),
            "y": vector(solution.y, self.m),
            "prim_inf_cert": vector(solution.prim_inf_cert, self.m),
            "dual_inf_cert": vector(solution.dual_inf_cert, self.n),
        }

    def close(self):
        library = self.library
        if self._solver:
            library.osqp_cleanup(self._solver)
            self._solver = ct.POINTER(OSQPSolver)()
        if self._settings:
            library.OSQPSettings_free(self._settings)
            self._settings = None
        if self._P_csc:
            library.OSQPCscMatrix_free(self._P_csc)
            self._P_csc = None
        if self._A_csc:
            library.OSQPCscMatrix_free(self._A_csc)
            self._A_csc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# C setting name -> Torch setting name, for the settings that both sides share.
_TORCH_NAMES = {
    "adaptive_rho_interval": "rho_update_interval",
    "adaptive_rho_tolerance": "rho_update_tolerance",
    "warm_starting": "warm_start",
}


def paired_settings(**c_overrides):
    """Return ``(c_settings, torch_settings)`` describing the same configuration."""

    c_settings = {
        "rho": 0.1,
        "sigma": 1e-6,
        "alpha": 1.6,
        "max_iter": 4000,
        "eps_abs": 1e-3,
        "eps_rel": 1e-3,
        "eps_prim_inf": 1e-4,
        "eps_dual_inf": 1e-4,
        "scaling": 10,
        "adaptive_rho": 1,
        "adaptive_rho_interval": 50,
        "adaptive_rho_fraction": 0.4,
        "adaptive_rho_tolerance": 5.0,
        "check_termination": 25,
        "check_dualgap": 1,
        "scaled_termination": 0,
        "warm_starting": 1,
        "polishing": 0,
        "polish_refine_iter": 3,
    }
    c_settings.update(c_overrides)
    torch_settings = {_TORCH_NAMES.get(k, k): v for k, v in c_settings.items()}
    for key in ("check_dualgap", "scaled_termination", "warm_start", "polishing"):
        torch_settings[key] = bool(torch_settings[key])
    torch_settings["return_info"] = True
    torch_settings["return_state"] = True
    return c_settings, torch_settings
