"""Test-only readout of the pinned OSQP C workspace (builtin algebra, QDLDL).

The layouts mirror ``include/private/types.h``, ``algebra/builtin/algebra_impl.h``
and ``algebra/_common/lin_sys/qdldl/qdldl_interface.h`` at the pinned commit for
the retained build (double / long long, profiling, printing, interrupt and
derivatives enabled, non-embedded).  Everything is copied out of C memory so the
values can be compared bit for bit with the Torch translation.
"""

from __future__ import annotations

import ctypes as ct

import numpy as np
from pinned_osqp_c_oracle import OSQPCscMatrix, OSQPFloat, OSQPInt

FP = ct.POINTER(OSQPFloat)
IP = ct.POINTER(OSQPInt)
CSC = ct.POINTER(OSQPCscMatrix)
FN = ct.c_void_p


class VecF(ct.Structure):
    _fields_ = [("values", FP), ("length", OSQPInt)]


class VecI(ct.Structure):
    _fields_ = [("values", IP), ("length", OSQPInt)]


class Mat(ct.Structure):
    _fields_ = [("csc", CSC), ("symmetry", ct.c_int)]


VF = ct.POINTER(VecF)
VI = ct.POINTER(VecI)
MP = ct.POINTER(Mat)


class Scaling(ct.Structure):
    _fields_ = [
        ("c", OSQPFloat), ("D", VF), ("E", VF),
        ("cinv", OSQPFloat), ("Dinv", VF), ("Einv", VF),
    ]


class Data(ct.Structure):
    _fields_ = [
        ("n", OSQPInt), ("m", OSQPInt), ("P", MP), ("A", MP),
        ("q", VF), ("l", VF), ("u", VF),
    ]


class Qdldl(ct.Structure):
    _fields_ = [
        ("type", ct.c_int),
        ("name", FN), ("solve", FN), ("update_settings", FN), ("warm_start", FN),
        ("adjoint_derivative", FN), ("free", FN),
        ("update_matrices", FN), ("update_rho_vec", FN),
        ("nthreads", OSQPInt),
        ("L", CSC), ("Dinv", FP), ("P", IP), ("bp", FP), ("sol", FP),
        ("rho_inv_vec", FP), ("sigma", OSQPFloat), ("rho_inv", OSQPFloat),
        ("polishing", OSQPInt), ("n", OSQPInt), ("m", OSQPInt),
        ("KKT", CSC), ("PtoKKT", IP), ("AtoKKT", IP), ("rhotoKKT", IP),
        ("D", FP), ("etree", IP), ("Lnz", IP),
        ("iwork", IP), ("bwork", ct.c_void_p), ("fwork", FP),
        ("adj", CSC),
    ]


class Workspace(ct.Structure):
    _fields_ = [
        ("data", ct.POINTER(Data)),
        ("linsys_solver", ct.POINTER(Qdldl)),
        ("pol", ct.c_void_p),
        ("rho_vec", VF), ("rho_inv_vec", VF),
        ("constr_type", VI),
        ("x", VF), ("y", VF), ("z", VF),
        ("xz_tilde", VF), ("xtilde_view", VF), ("ztilde_view", VF),
        ("x_prev", VF), ("z_prev", VF),
        ("Ax", VF), ("Px", VF), ("Aty", VF),
        ("xtPx", OSQPFloat), ("qtx", OSQPFloat), ("SC", OSQPFloat),
        ("scaled_dual_gap", OSQPFloat),
        ("delta_y", VF), ("Atdelta_y", VF),
        ("delta_x", VF), ("Pdelta_x", VF), ("Adelta_x", VF),
        ("D_temp", VF), ("D_temp_A", VF), ("E_temp", VF),
        ("scaling", ct.POINTER(Scaling)),
        ("scaled_prim_res", OSQPFloat), ("scaled_dual_res", OSQPFloat),
        ("rho_inv", OSQPFloat),
    ]


def _vf(pointer):
    if not pointer:
        return None
    vector = pointer.contents
    if int(vector.length) == 0 or not vector.values:
        return np.empty(0, dtype=np.float64)
    return np.ctypeslib.as_array(vector.values, shape=(int(vector.length),)).copy()


def _vi(pointer):
    if not pointer:
        return None
    vector = pointer.contents
    if int(vector.length) == 0 or not vector.values:
        return np.empty(0, dtype=np.int64)
    return np.ctypeslib.as_array(vector.values, shape=(int(vector.length),)).copy()


def _raw(pointer, size, dtype=None):
    if not pointer or size == 0:
        return np.empty(0, dtype=dtype or np.float64)
    return np.ctypeslib.as_array(pointer, shape=(int(size),)).copy()


def _csc(pointer):
    if not pointer:
        return None
    matrix = pointer.contents
    n = int(matrix.n)
    p = _raw(matrix.p, n + 1, np.int64)
    nnz = int(p[-1])
    return {
        "m": int(matrix.m), "n": n, "p": p,
        "i": _raw(matrix.i, nnz, np.int64), "x": _raw(matrix.x, nnz),
    }


def csc_to_dense(csc):
    dense = np.zeros((csc["m"], csc["n"]))
    for column in range(csc["n"]):
        for k in range(csc["p"][column], csc["p"][column + 1]):
            dense[csc["i"][k], column] = csc["x"][k]
    return dense


def read_workspace(solver):
    """Copy the scaled problem data, iterates and factors out of ``solver->work``."""

    work = ct.cast(solver.contents.work, ct.POINTER(Workspace)).contents
    data = work.data.contents
    n, m = int(data.n), int(data.m)
    out = {
        "n": n, "m": m,
        "P": _csc(data.P.contents.csc), "A": _csc(data.A.contents.csc),
        "q": _vf(data.q), "l": _vf(data.l), "u": _vf(data.u),
        "rho_vec": _vf(work.rho_vec), "rho_inv_vec": _vf(work.rho_inv_vec),
        "constr_type": _vi(work.constr_type),
        "x": _vf(work.x), "y": _vf(work.y), "z": _vf(work.z),
        "x_prev": _vf(work.x_prev), "z_prev": _vf(work.z_prev),
        "xz_tilde": _vf(work.xz_tilde),
        "Ax": _vf(work.Ax), "Px": _vf(work.Px), "Aty": _vf(work.Aty),
        "delta_x": _vf(work.delta_x), "delta_y": _vf(work.delta_y),
        "xtPx": float(work.xtPx), "qtx": float(work.qtx), "SC": float(work.SC),
        "scaled_dual_gap": float(work.scaled_dual_gap),
        "scaled_prim_res": float(work.scaled_prim_res),
        "scaled_dual_res": float(work.scaled_dual_res),
    }
    if work.scaling:
        scaling = work.scaling.contents
        out.update(
            c=float(scaling.c), cinv=float(scaling.cinv),
            D=_vf(scaling.D), E=_vf(scaling.E),
            Dinv=_vf(scaling.Dinv), Einv=_vf(scaling.Einv),
        )
    lin = work.linsys_solver.contents
    size = n + m
    assert int(lin.n) == n and int(lin.m) == m, (int(lin.n), int(lin.m), n, m)
    out["linsys"] = {
        "perm": _raw(lin.P, size, np.int64),
        "L": _csc(lin.L),
        "D": _raw(lin.D, size),
        "Dinv": _raw(lin.Dinv, size),
        "KKT": _csc(lin.KKT),
        "etree": _raw(lin.etree, size, np.int64),
        "Lnz": _raw(lin.Lnz, size, np.int64),
        "sigma": float(lin.sigma),
    }
    return out
