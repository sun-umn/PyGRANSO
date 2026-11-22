"""Test-only bridge to the pinned double/long OSQP C oracle."""

from __future__ import annotations

import ctypes as ct
import hashlib
import os
from pathlib import Path

import numpy as np

OSQP_COMMIT = "1572ae068e9ce9ca723cf8223548ade1ff7acc29"
OSQP_DLL_SHA256 = "97d14c66f40b471e3220454374d3bcfc429c64e6297b0062b6f60a0264e043c9"

OSQPInt = ct.c_longlong
OSQPFloat = ct.c_double


class OSQPSettings(ct.Structure):
    _fields_ = [
        ("device", OSQPInt),
        ("linsys_solver", ct.c_int),
        ("allocate_solution", OSQPInt),
        ("verbose", OSQPInt),
        ("profiler_level", OSQPInt),
        ("warm_starting", OSQPInt),
        ("scaling", OSQPInt),
        ("polishing", OSQPInt),
        ("rho", OSQPFloat),
        ("rho_is_vec", OSQPInt),
        ("sigma", OSQPFloat),
        ("alpha", OSQPFloat),
        ("cg_max_iter", OSQPInt),
        ("cg_tol_reduction", OSQPInt),
        ("cg_tol_fraction", OSQPFloat),
        ("cg_precond", ct.c_int),
        ("adaptive_rho", OSQPInt),
        ("adaptive_rho_interval", OSQPInt),
        ("adaptive_rho_fraction", OSQPFloat),
        ("adaptive_rho_tolerance", OSQPFloat),
        ("max_iter", OSQPInt),
        ("eps_abs", OSQPFloat),
        ("eps_rel", OSQPFloat),
        ("eps_prim_inf", OSQPFloat),
        ("eps_dual_inf", OSQPFloat),
        ("scaled_termination", OSQPInt),
        ("check_termination", OSQPInt),
        ("check_dualgap", OSQPInt),
        ("time_limit", OSQPFloat),
        ("delta", OSQPFloat),
        ("polish_refine_iter", OSQPInt),
    ]


class OSQPInfo(ct.Structure):
    _fields_ = [
        ("status", ct.c_char * 32),
        ("status_val", OSQPInt),
        ("status_polish", OSQPInt),
        ("obj_val", OSQPFloat),
        ("dual_obj_val", OSQPFloat),
        ("prim_res", OSQPFloat),
        ("dual_res", OSQPFloat),
        ("duality_gap", OSQPFloat),
        ("iter", OSQPInt),
        ("rho_updates", OSQPInt),
        ("rho_estimate", OSQPFloat),
        ("setup_time", OSQPFloat),
        ("solve_time", OSQPFloat),
        ("update_time", OSQPFloat),
        ("polish_time", OSQPFloat),
        ("run_time", OSQPFloat),
        ("primdual_int", OSQPFloat),
        ("rel_kkt_error", OSQPFloat),
    ]


class OSQPSolution(ct.Structure):
    _fields_ = [
        ("x", ct.POINTER(OSQPFloat)),
        ("y", ct.POINTER(OSQPFloat)),
        ("prim_inf_cert", ct.POINTER(OSQPFloat)),
        ("dual_inf_cert", ct.POINTER(OSQPFloat)),
    ]


class OSQPSolver(ct.Structure):
    _fields_ = [
        ("settings", ct.POINTER(OSQPSettings)),
        ("solution", ct.POINTER(OSQPSolution)),
        ("info", ct.POINTER(OSQPInfo)),
        ("work", ct.c_void_p),
    ]


class OSQPCscMatrix(ct.Structure):
    _fields_ = [
        ("m", OSQPInt),
        ("n", OSQPInt),
        ("p", ct.POINTER(OSQPInt)),
        ("i", ct.POINTER(OSQPInt)),
        ("x", ct.POINTER(OSQPFloat)),
        ("nzmax", OSQPInt),
        ("nz", OSQPInt),
        ("owned", OSQPInt),
    ]


def pinned_dll_path() -> Path:
    return Path(os.environ.get("PYGRANSO_OSQP_DLL", ""))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pinned_library(path: Path | None = None):
    path = pinned_dll_path() if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Pinned OSQP DLL is unavailable: {path}")
    actual_hash = _sha256(path)
    if actual_hash != OSQP_DLL_SHA256:
        raise RuntimeError(
            f"Pinned OSQP DLL hash mismatch: expected {OSQP_DLL_SHA256}, "
            f"got {actual_hash}."
        )

    library = ct.CDLL(str(path))
    library.OSQPCscMatrix_new.restype = ct.POINTER(OSQPCscMatrix)
    library.OSQPCscMatrix_new.argtypes = [
        OSQPInt,
        OSQPInt,
        OSQPInt,
        ct.POINTER(OSQPFloat),
        ct.POINTER(OSQPInt),
        ct.POINTER(OSQPInt),
    ]
    library.OSQPCscMatrix_free.argtypes = [ct.POINTER(OSQPCscMatrix)]
    library.OSQPSettings_new.restype = ct.POINTER(OSQPSettings)
    library.OSQPSettings_free.argtypes = [ct.POINTER(OSQPSettings)]
    library.osqp_version.restype = ct.c_char_p
    library.osqp_setup.restype = OSQPInt
    library.osqp_setup.argtypes = [
        ct.POINTER(ct.POINTER(OSQPSolver)),
        ct.POINTER(OSQPCscMatrix),
        ct.POINTER(OSQPFloat),
        ct.POINTER(OSQPCscMatrix),
        ct.POINTER(OSQPFloat),
        ct.POINTER(OSQPFloat),
        OSQPInt,
        OSQPInt,
        ct.POINTER(OSQPSettings),
    ]
    library.osqp_solve.restype = OSQPInt
    library.osqp_solve.argtypes = [ct.POINTER(OSQPSolver)]
    library.osqp_cleanup.restype = OSQPInt
    library.osqp_cleanup.argtypes = [ct.POINTER(OSQPSolver)]
    return library


def _as_float(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def _as_index(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def _pointer(array: np.ndarray, scalar_type):
    return array.ctypes.data_as(ct.POINTER(scalar_type))


def dense_to_csc(matrix, *, upper_triangular: bool = False):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("CSC input must be a two-dimensional matrix.")
    if upper_triangular:
        matrix = np.triu(matrix)
    rows = []
    values = []
    pointers = [0]
    for column in range(matrix.shape[1]):
        indices = np.flatnonzero(matrix[:, column])
        rows.extend(indices.tolist())
        values.extend(matrix[indices, column].tolist())
        pointers.append(len(rows))
    return _as_float(values), _as_index(rows), _as_index(pointers)


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
        ct.POINTER(OSQPFloat),
        ct.POINTER(OSQPInt),
        OSQPInt,
        ct.POINTER(OSQPFloat),
        ct.POINTER(OSQPInt),
        OSQPInt,
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
            self.n,
            self.n,
            self._P[0].size,
            _pointer(self._P[0], OSQPFloat),
            _pointer(self._P[1], OSQPInt),
            _pointer(self._P[2], OSQPInt),
        )
        self._A_csc = library.OSQPCscMatrix_new(
            self.m,
            self.n,
            self._A[0].size,
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

        arrays = [
            None if v is None else _as_float(v).reshape(-1).copy() for v in (q, l, u)
        ]
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

    def problem_vectors(self):
        # First workspace field in the pinned builtin ABI is OSQPData *data.
        data = ct.cast(self._solver.contents.work, ct.POINTER(ct.POINTER(_Data)))[
            0
        ].contents
        vectors = {}
        for key in ("q", "l", "u"):
            vector = getattr(data, key).contents
            vectors[key] = np.ctypeslib.as_array(
                vector.values, shape=(vector.length,)
            ).copy()
        return vectors

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


class _Vector(ct.Structure):
    _fields_ = [("values", ct.POINTER(OSQPFloat)), ("length", OSQPInt)]


class _Data(ct.Structure):
    _fields_ = [
        ("n", OSQPInt),
        ("m", OSQPInt),
        ("P", ct.c_void_p),
        ("A", ct.c_void_p),
        ("q", ct.POINTER(_Vector)),
        ("l", ct.POINTER(_Vector)),
        ("u", ct.POINTER(_Vector)),
    ]
