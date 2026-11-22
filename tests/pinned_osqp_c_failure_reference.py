"""Native CPU callback evidence from the retained, hash-checked DLL."""

import ctypes as ct
from contextlib import contextmanager

import numpy as np
from pinned_osqp_c_internals import Qdldl, VecF, Workspace, csc_to_dense, read_workspace
from pinned_osqp_c_oracle import OSQPFloat, OSQPInt, OSQPSettings, OSQPSolver
from pinned_osqp_c_session import PinnedOSQPSession, paired_settings


class WorkspaceEnd(ct.Structure):
    _fields_ = Workspace._fields_ + [
        ("timer", ct.c_void_p),
        ("first_run", OSQPInt),
        ("clear_update_time", OSQPInt),
        ("rho_update_from_solve", OSQPInt),
        ("summary_printed", OSQPInt),
        ("derivative_data", ct.c_void_p),
        ("rho_updated", OSQPInt),
        ("last_rel_kkt", OSQPFloat),
    ]


@contextmanager
def native(problem, **overrides):
    settings, _ = paired_settings(
        **(
            {"scaling": 0, "adaptive_rho": 0, "eps_abs": 1e-8, "eps_rel": 1e-8}
            | overrides
        )
    )
    with PinnedOSQPSession(*problem, **settings) as c:
        assert c.setup_code == 0, c.setup_code
        c.library.osqp_update_rho.argtypes = [ct.POINTER(OSQPSolver), OSQPFloat]
        c.library.osqp_update_rho.restype = OSQPInt
        c.library.osqp_update_settings.argtypes = [
            ct.POINTER(OSQPSolver),
            ct.POINTER(OSQPSettings),
        ]
        c.library.osqp_update_settings.restype = OSQPInt
        yield c


def polishing(c, value):
    settings = OSQPSettings.from_buffer_copy(c._solver.contents.settings.contents)
    settings.polishing = value
    assert c.library.osqp_update_settings(c._solver, ct.byref(settings)) == 0


def step(c, function):
    ret = function()
    solver = c._solver.contents
    info = solver.info.contents
    sol = solver.solution.contents
    w = read_workspace(c._solver)
    full = ct.cast(solver.work, ct.POINTER(WorkspaceEnd)).contents
    return {
        "rc": ret["solve_code"] if isinstance(ret, dict) else int(ret),
        **{
            k: getattr(info, k)
            for k in (
                "status_val",
                "iter",
                "status_polish",
                "rho_updates",
                "rho_estimate",
                "obj_val",
                "dual_obj_val",
                "duality_gap",
                "prim_res",
                "dual_res",
                "rel_kkt_error",
            )
        },
        "rho": float(solver.settings.contents.rho),
        "history": full.last_rel_kkt,
        "rho_updated": full.rho_updated,
        **{
            k: w[k]
            for k in (
                "x",
                "z",
                "y",
                "q",
                "l",
                "u",
                "rho_vec",
                "rho_inv_vec",
                "constr_type",
            )
        },
        "P": csc_to_dense(w["P"]),
        "A": csc_to_dense(w["A"]),
        "KKT_x": w["linsys"]["KKT"]["x"],
        "D": w["linsys"]["D"],
        "Dinv": w["linsys"]["Dinv"],
        "public_x": np.ctypeslib.as_array(sol.x, shape=(c.n,)).copy(),
        "public_y": np.ctypeslib.as_array(sol.y, shape=(c.m,)).copy(),
    }


def matrix(scaling, bad):
    p = [[2.0]]
    with native((p, [-1.0], [[1.0]], [-1.0], [1.0]), scaling=scaling, sigma=0.5) as c:
        return dict(
            initial=step(c, c.solve),
            failed=step(c, lambda: c.update_matrices(P=[[bad]])),
            repair=step(c, lambda: c.update_matrices(P=p)),
            retry=step(c, c.solve),
        )


def rho(bounds):
    with native(
        ([[-1.5]], [0.0], [[1.0]], [0.0 if bounds else -1.0], [0.0 if bounds else 1.0]),
        sigma=0.5,
        rho=1.0 if bounds else 2.0,
    ) as c:
        if bounds:
            fail, fix = lambda: c.update(q=[0.25], u=[1.0]), lambda: c.update(u=[0.0])
        else:
            fail, fix = (
                lambda: c.library.osqp_update_rho(c._solver, 1.0),
                lambda: c.library.osqp_update_rho(c._solver, 2.0),
            )
        return dict(
            initial=step(c, c.solve),
            failed=step(c, fail),
            repair=step(c, fix),
            retry=step(c, c.solve),
        )


def polish():
    with native(([[-2.0]], [-0.25], [[1.0]], [0.0], [0.0]), sigma=1.0) as c:
        first = step(c, c.solve)
        changed = step(c, lambda: c.update(q=[0.0]))
        polishing(c, 1)
        failed = step(c, c.solve)
        polishing(c, 0)
        return dict(
            initial=first, changed=changed, failed=failed, retry=step(c, c.solve)
        )


def adapt():
    problem = (
        [[4.0, 1.0], [1.0, 2.0]],
        [1.0, 1.0],
        [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        [1.0, 0.0, 0.0, -1e30],
        [1.0, 0.7, 0.7, 1e30],
    )
    signature = ct.CFUNCTYPE(OSQPInt, ct.POINTER(Qdldl), ct.POINTER(VecF), OSQPFloat)
    with native(
        problem, rho=1e-4, adaptive_rho=1, adaptive_rho_interval=25, max_iter=2000
    ) as c:
        initial = step(c, c.solve)
        changed = step(c, lambda: c.update(q=[0.0, 1.0]))
        lin = ct.cast(
            c._solver.contents.work, ct.POINTER(Workspace)
        ).contents.linsys_solver
        original = lin.contents.update_rho_vec
        called = []

        @signature
        def fail(_solver, vec, candidate):
            called.append(
                {
                    "rho": candidate,
                    "rho_vec": vec.contents.values[: c.m],
                    "count_on_entry": c._solver.contents.info.contents.rho_updates,
                }
            )
            return 1

        lin.contents.update_rho_vec = ct.cast(fail, ct.c_void_p).value
        try:
            failed = step(c, c.solve)
        finally:
            lin.contents.update_rho_vec = original
        assert len(called) == 1
        repair = step(c, lambda: c.library.osqp_update_rho(c._solver, failed["rho"]))
        return dict(
            initial=initial,
            changed=changed,
            failed=failed,
            callback=called,
            repair=repair,
            retry=step(c, c.solve),
        )


def pack(value):
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    if isinstance(value, (np.ndarray, list, tuple)):
        return [pack(v) for v in value]
    if isinstance(value, np.generic):
        return pack(value.item())
    return value.hex() if isinstance(value, float) else value
