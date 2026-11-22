"""Raw binary64 comparisons at solver boundaries using the pinned builtin DLL."""

import ctypes as ct
import math

import numpy as np
import pytest
import torch
from pinned_osqp_c_oracle import (
    PinnedOSQPSession,
    _Vector,
    paired_settings,
    pinned_dll_path,
)

from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchOSQP import _OSQPCallbackError, solve_torch_osqp_direct

pytestmark = pytest.mark.skipif(
    not pinned_dll_path().is_file(), reason="requires the pinned double native DLL"
)


class _WorkspaceIterates(ct.Structure):
    # Prefix of OSQPWorkspace from include/private/types.h at the pinned commit.
    _fields_ = [
        (name, ct.c_void_p)
        for name in ("data", "linsys_solver", "pol", "rho_vec", "rho_inv_vec", "constr_type")
    ] + [(name, ct.POINTER(_Vector)) for name in ("x", "y", "z")]


def _compare_iterates(session, workspace):
    native = ct.cast(session._solver.contents.work, ct.POINTER(_WorkspaceIterates)).contents
    for name in ("x", "z", "y"):
        vector = getattr(native, name).contents
        expected = np.ctypeslib.as_array(vector.values, shape=(vector.length,))
        _same_bits(workspace.work_iterates[name], expected, f"working {name}")


def _same_bits(actual, expected, field):
    actual, expected = (np.asarray(value, dtype="<f8") for value in (actual, expected))
    assert actual.shape == expected.shape, field
    assert actual.tobytes() == expected.tobytes(), (
        f"{field}: Torch {actual.tobytes().hex()} vs C {expected.tobytes().hex()}"
    )


def _compare_public(native, x, info):
    for c_name, torch_name in (
        ("status_val", "status_val"),
        ("status_polish", "status_polish"),
        ("iter", "admm_iterations"),
        ("rho_updates", "rho_updates"),
    ):
        assert info[torch_name] == native[c_name], c_name
    for c_name, torch_name in (
        ("rho", "rho_bar"),
        ("rho_estimate", "rho_estimate"),
        ("objective", "objective"),
        ("dual_objective", "dual_objective"),
        ("duality_gap", "duality_gap"),
        ("primal_residual", "primal_residual"),
        ("dual_residual", "dual_residual"),
        ("rel_kkt_error", "rel_kkt_error"),
    ):
        _same_bits(info[torch_name], native[c_name], c_name)
    _same_bits(x.reshape(-1), native["x"], "public x")
    _same_bits(info["public_y"], native["y"], "public y")


def _solve(problem, settings, workspace):
    return solve_torch_osqp_direct(
        *(torch.as_tensor(vector, dtype=torch.float64) for vector in problem),
        settings,
        workspace,
    )


def test_signed_zero_bounds_and_iterates_keep_the_native_projection_bits():
    P = A = np.eye(2)
    problem = (P, [-0.0, 0.0], A, [-0.0, 0.0], [0.0, -0.0])
    c_settings, torch_settings = paired_settings(scaling=0, check_termination=1)
    workspace = TorchOSQPWorkspace()
    with PinnedOSQPSession(*problem, **c_settings) as session:
        assert session.setup_code == 0
        native = session.solve()
        assert native["solve_code"] == 0
        x, info = _solve(problem, torch_settings, workspace)
        for key, submitted, retained in zip(
            ("l", "u"), problem[-2:], workspace.status_problem[-2:], strict=True,
        ):
            _same_bits(retained, submitted, f"submitted {key}")
            _same_bits(retained, session.problem_vectors()[key], f"installed {key}")
        _compare_iterates(session, workspace)
        _same_bits(workspace.work_iterates["z"], [0.0, -0.0], "projection tie")
    _compare_public(native, x, info)


def test_ieee_infinite_bound_reaches_the_native_iteration_cap_with_nan_gap():
    P = A = np.eye(2)
    problem = (P, [1.0, -1.0], A, [0.0, 0.0], [np.inf, 2.0])
    c_settings, torch_settings = paired_settings(max_iter=75)
    workspace = TorchOSQPWorkspace()
    with PinnedOSQPSession(*problem, **c_settings) as session:
        assert session.setup_code == 0
        native = session.solve()
        assert native["solve_code"] == 0
        x, info = _solve(problem, torch_settings, workspace)
        _compare_iterates(session, workspace)
    assert native["status_val"] == 7 and native["iter"] == c_settings["max_iter"]
    assert info["status"] == "max_iter_reached"
    assert math.isnan(native["duality_gap"]) and math.isnan(info["duality_gap"])
    _compare_public(native, x, info)


def test_infeasible_solve_retains_native_public_dual_buffer():
    problem = ([[1.0]], [0.0], [[1.0], [1.0]], [1.0, -1e30], [1e30, 0.0])
    c_settings, torch_settings = paired_settings(scaling=0)
    workspace = TorchOSQPWorkspace()
    with PinnedOSQPSession(*problem, **c_settings) as session:
        assert session.setup_code == 0
        native = session.solve()
        assert native["solve_code"] == 0 and native["status_val"] == 3
        x, info = _solve(problem, torch_settings, workspace)
    _compare_public(native, x, info)
    _same_bits(workspace.solver_info["public_y"], native["y"], "retained public y")
    _same_bits(info["prim_inf_cert"], native["prim_inf_cert"], "primal certificate")


def test_polish_failure_and_retry_leave_the_native_public_and_working_state():
    P, A, l, u = np.diag([1.0, -0.05]), np.eye(2), -np.ones(2), np.ones(2)
    q = np.array([5.0, 0.0])
    c_settings, torch_settings = paired_settings(polishing=1)
    workspace = TorchOSQPWorkspace()
    with PinnedOSQPSession(P, q, A, l, u, **c_settings) as session:
        assert session.setup_code == 0
        for _ in range(2):
            native = session.solve()
            assert native["solve_code"] == 4 and native["status_polish"] == -2
            with pytest.raises(_OSQPCallbackError) as raised:
                _solve((P, q, A, l, u), torch_settings, workspace)
            failure = raised.value
            assert (failure.stage, failure.exit_code) == ("polishing", native["solve_code"])
            _compare_public(native, failure.info["public_x"], failure.info)
            _compare_iterates(session, workspace)

        q = np.array([5.0, 0.5])
        assert session.update(q=q) == 0
        native = session.solve()
        assert native["solve_code"] == 0 and native["status_polish"] == 1
        x, info = _solve((P, q, A, l, u), torch_settings, workspace)
        _compare_iterates(session, workspace)
    _compare_public(native, x, info)
