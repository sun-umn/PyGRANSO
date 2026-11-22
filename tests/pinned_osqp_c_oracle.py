"""Test-only bridge to the pinned double/long OSQP C oracle."""

from __future__ import annotations

import ctypes as ct
import hashlib
from pathlib import Path

import numpy as np

OSQP_COMMIT = "1572ae068e9ce9ca723cf8223548ade1ff7acc29"
OSQP_DLL_SHA256 = "97d14c66f40b471e3220454374d3bcfc429c64e6297b0062b6f60a0264e043c9"
OSQP_DLL_RELATIVE_PATH = Path(
    "tmp/phase54-osqp-c-build-release-20260904/out/osqp.dll"
)

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


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def pinned_dll_path() -> Path:
    return repository_root() / OSQP_DLL_RELATIVE_PATH


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


def default_settings(library=None) -> dict[str, int | float | str]:
    library = load_pinned_library() if library is None else library
    settings = library.OSQPSettings_new()
    if not settings:
        raise MemoryError("Pinned OSQP could not allocate default settings.")
    try:
        values = settings.contents
        return {
            "version": library.osqp_version().decode(),
            "rho": float(values.rho),
            "sigma": float(values.sigma),
            "alpha": float(values.alpha),
            "max_iter": int(values.max_iter),
            "eps_abs": float(values.eps_abs),
            "eps_rel": float(values.eps_rel),
            "eps_prim_inf": float(values.eps_prim_inf),
            "eps_dual_inf": float(values.eps_dual_inf),
            "scaling": int(values.scaling),
            "polishing": int(values.polishing),
            "adaptive_rho": int(values.adaptive_rho),
            "adaptive_rho_interval": int(values.adaptive_rho_interval),
            "adaptive_rho_fraction": float(values.adaptive_rho_fraction),
            "adaptive_rho_tolerance": float(values.adaptive_rho_tolerance),
            "check_termination": int(values.check_termination),
            "check_dualgap": int(values.check_dualgap),
            "time_limit": float(values.time_limit),
            "delta": float(values.delta),
            "polish_refine_iter": int(values.polish_refine_iter),
        }
    finally:
        library.OSQPSettings_free(settings)


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


def solve_dense_qp(
    P,
    q,
    A,
    l,
    u,
    *,
    settings_overrides: dict[str, int | float] | None = None,
    library=None,
):
    library = load_pinned_library() if library is None else library
    P = np.asarray(P, dtype=np.float64)
    q = _as_float(q).reshape(-1)
    A = np.asarray(A, dtype=np.float64)
    l = _as_float(l).reshape(-1)
    u = _as_float(u).reshape(-1)
    n = q.size
    m = l.size
    if P.shape != (n, n) or A.shape != (m, n) or u.size != m:
        raise ValueError("Pinned oracle received incompatible dense QP shapes.")

    P_values, P_rows, P_pointers = dense_to_csc(P, upper_triangular=True)
    A_values, A_rows, A_pointers = dense_to_csc(A)
    P_csc = library.OSQPCscMatrix_new(
        n,
        n,
        P_values.size,
        _pointer(P_values, OSQPFloat),
        _pointer(P_rows, OSQPInt),
        _pointer(P_pointers, OSQPInt),
    )
    A_csc = library.OSQPCscMatrix_new(
        m,
        n,
        A_values.size,
        _pointer(A_values, OSQPFloat),
        _pointer(A_rows, OSQPInt),
        _pointer(A_pointers, OSQPInt),
    )
    settings = library.OSQPSettings_new()
    if not P_csc or not A_csc or not settings:
        if settings:
            library.OSQPSettings_free(settings)
        if P_csc:
            library.OSQPCscMatrix_free(P_csc)
        if A_csc:
            library.OSQPCscMatrix_free(A_csc)
        raise MemoryError("Pinned OSQP could not allocate oracle inputs.")

    settings.contents.verbose = 0
    for name, value in (settings_overrides or {}).items():
        if name not in dict(OSQPSettings._fields_):
            raise ValueError(f"Unknown pinned OSQP setting {name!r}.")
        setattr(settings.contents, name, value)

    solver = ct.POINTER(OSQPSolver)()
    setup_code = library.osqp_setup(
        ct.byref(solver),
        P_csc,
        _pointer(q, OSQPFloat),
        A_csc,
        _pointer(l, OSQPFloat),
        _pointer(u, OSQPFloat),
        m,
        n,
        settings,
    )
    try:
        if setup_code:
            return {"setup_code": int(setup_code)}
        solve_code = library.osqp_solve(solver)
        info = solver.contents.info.contents
        solution = solver.contents.solution.contents
        x = np.ctypeslib.as_array(solution.x, shape=(n,)).copy()
        y = np.ctypeslib.as_array(solution.y, shape=(m,)).copy()
        Ax = A @ x
        z = np.clip(Ax, l, u)
        Px = P @ x
        ATy = A.T @ y
        primal_residual = float(np.max(np.abs(Ax - z))) if m else 0.0
        dual_residual = float(np.max(np.abs(Px + q + ATy)))
        eps_primal = float(
            settings.contents.eps_abs
            + settings.contents.eps_rel
            * max(
                float(np.max(np.abs(Ax))) if m else 0.0,
                float(np.max(np.abs(z))) if m else 0.0,
            )
        )
        eps_dual = float(
            settings.contents.eps_abs
            + settings.contents.eps_rel
            * max(
                float(np.max(np.abs(Px))),
                float(np.max(np.abs(ATy))),
                float(np.max(np.abs(q))),
            )
        )
        return {
            "setup_code": int(setup_code),
            "solve_code": int(solve_code),
            "status": info.status.split(b"\0", 1)[0].decode(),
            "status_val": int(info.status_val),
            "status_polish": int(info.status_polish),
            "iter": int(info.iter),
            "rho_updates": int(info.rho_updates),
            "rho_estimate": float(info.rho_estimate),
            "rel_kkt_error": float(info.rel_kkt_error),
            "adaptive_rho": int(solver.contents.settings.contents.adaptive_rho),
            "adaptive_rho_interval": int(
                solver.contents.settings.contents.adaptive_rho_interval
            ),
            "adaptive_rho_fraction": float(
                solver.contents.settings.contents.adaptive_rho_fraction
            ),
            "setup_time": float(info.setup_time),
            "run_time": float(info.run_time),
            "objective": float(info.obj_val),
            "reported_primal_residual": float(info.prim_res),
            "reported_dual_residual": float(info.dual_res),
            "independent_primal_residual": primal_residual,
            "independent_dual_residual": dual_residual,
            "eps_primal": eps_primal,
            "eps_dual": eps_dual,
            "x": x,
            "y": y,
        }
    finally:
        if solver:
            library.osqp_cleanup(solver)
        library.OSQPSettings_free(settings)
        library.OSQPCscMatrix_free(P_csc)
        library.OSQPCscMatrix_free(A_csc)


def solve_saved_pygranso_problem(
    problem,
    *,
    settings_overrides: dict[str, int | float] | None = None,
    library=None,
):
    H, q, equality, rhs, lower, upper = problem
    H = H.detach().cpu().numpy()
    q = q.detach().cpu().reshape(-1).numpy()
    equality = equality.detach().cpu().numpy()
    rhs = np.asarray(rhs.detach().cpu()).reshape(-1)
    lower = lower.detach().cpu().reshape(-1).numpy()
    upper = upper.detach().cpu().reshape(-1).numpy()
    n = H.shape[0]
    A = np.concatenate((equality, np.eye(n)), axis=0)
    l = np.concatenate((rhs, lower))
    u = np.concatenate((rhs, upper))
    return solve_dense_qp(
        H,
        q,
        A,
        l,
        u,
        settings_overrides=settings_overrides,
        library=library,
    )
