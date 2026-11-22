"""Observable counterparts of the pinned basic_qp and update_matrices tests.

The standalone native runner covers C-only APIs. Here all three C index
encodings are exercised against the same tensor matrix submission.
"""

import ctypes as ct

import numpy as np
import pytest
import torch
from c_parity_support import PinnedOSQPSession, paired_settings
from pinned_osqp_c_internals import read_workspace
from pinned_osqp_c_oracle import OSQPFloat, OSQPInt, pinned_dll_path
from test_osqp_backend_agreement import (
    _pinned_basic_qp2_fixture,
    _pinned_basic_qp_fixture,
    _pinned_update_matrices_fixture,
)

from pygranso.private.osqpTorchAdapter import _normalize_options
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchLinearSolve import TorchLinearSolveError
from pygranso.private.torchOSQP import solve_torch_osqp_direct


def native_data(problem):
    return {
        k: np.nan_to_num(problem[k].numpy(), posinf=1e30, neginf=-1e30)
        for k in ("P", "q", "A", "l", "u")
    }


def solve(data, device, options, workspace):
    return solve_torch_osqp_direct(
        *(
            torch.tensor(data[k], device=device, dtype=torch.float64)
            for k in ("P", "q", "A", "l", "u")
        ),
        options,
        workspace,
    )


@pytest.fixture
def oracle():
    if not pinned_dll_path().is_file():
        pytest.skip("live paired check requires the pinned double native DLL")
    return PinnedOSQPSession


@pytest.mark.parametrize(
    "field,value,option",
    [
        ("max_iter", -1, "max_iter"),
        ("max_iter", 0, "max_iter"),
        ("eps_abs", -1.0, "eps_abs"),
        ("eps_abs", -1.1, "eps_abs"),
        ("eps_rel", -1.0, "eps_rel"),
        ("eps_rel", -0.1, "eps_rel"),
        ("eps_prim_inf", -0.1, "eps_prim_inf"),
        ("eps_dual_inf", -0.1, "eps_dual_inf"),
        ("eps_dual_inf", 0.0, "eps_dual_inf"),
        ("alpha", 2.0, "alpha"),
        ("warm_starting", -1, "warm_start"),
        ("warm_starting", 5, "warm_start"),
        ("scaled_termination", 2, "scaled_termination"),
        ("check_termination", -1, "check_termination"),
        ("delta", 0.0, "polish_delta"),
        ("delta", -1.1, "polish_delta"),
        ("polishing", 2, "polishing"),
        ("polish_refine_iter", -1, "polish_refine_iter"),
        ("polish_refine_iter", -3, "polish_refine_iter"),
        ("verbose", 2, "verbose"),
        ("scaling", -1, "scaling"),
        ("adaptive_rho", 9, "adaptive_rho"),
        ("adaptive_rho_interval", -1, "rho_update_interval"),
        ("adaptive_rho_fraction", -1.5, "adaptive_rho_fraction"),
        ("adaptive_rho_tolerance", 0.5, "rho_update_tolerance"),
        ("rho", 0, "rho"),
        ("sigma", -0.1, "sigma"),
        ("time_limit", -0.2, "time_limit"),
    ],
)
def test_invalid_setting(oracle, field, value, option):
    # Both validators execute before the first numerical or device-specific step.
    problem = native_data(_pinned_basic_qp_fixture())
    with oracle(**problem, **{field: value}) as session:
        assert session.setup_code == 2
    with pytest.raises(ValueError):
        _normalize_options({"settings": {option: value}}, torch.float64)


@pytest.mark.parametrize("change", ["q", "l", "u", "flipped", "bad-l", "bad-u", "both"])
def test_vector_update(oracle, available_device, change):
    fixture = _pinned_basic_qp_fixture()
    data = native_data(fixture)
    new = {
        k: np.nan_to_num(fixture[k + "_new"].numpy(), posinf=1e30, neginf=-1e30)
        for k in ("q", "l", "u")
    }
    changes = {
        "q": {"q": new["q"]},
        "l": {"l": new["l"], "u": data["u"]},
        "u": {"l": data["l"], "u": new["u"]},
        "flipped": {"l": new["u"], "u": new["l"]},
        "bad-l": {"l": new["u"]},
        "bad-u": {"u": new["l"]},
        "both": {"l": new["l"], "u": new["u"]},
    }[change]
    invalid = change in {"flipped", "bad-l", "bad-u"}
    c, t = paired_settings(
        max_iter=200,
        scaling=0,
        polishing=1,
        warm_starting=0,
        eps_abs=1e-5,
        eps_rel=1e-5,
    )
    workspace = TorchOSQPWorkspace()
    with oracle(**data, **c) as session:
        assert session.setup_code == 0
        # Torch establishes a retained workspace by solving once. Apply the same
        # preparation on the C side for the additional post-update comparison.
        before = session.solve()
        _, info = solve(data, available_device, t, workspace)
        assert info["status_val"] == before["status_val"]
        committed = workspace.last_info
        assert session.update(**changes) == int(invalid)
        installed = read_workspace(session._solver)
        target = data if invalid else data | changes
        for key in ("q", "l", "u"):
            np.testing.assert_array_equal(installed[key], target[key])
        if invalid:
            with pytest.raises(ValueError, match="lower bound"):
                solve(data | changes, available_device, t, workspace)
            assert workspace.last_info is committed
        reference = session.solve()
        x, info = solve(target, available_device, t, workspace)
        assert info["status_val"] == reference["status_val"]
        np.testing.assert_allclose(x.cpu().flatten(), reference["x"], atol=1e-4, rtol=0)
        np.testing.assert_allclose(
            info["state"]["y"].cpu(), reference["y"], atol=1e-4, rtol=0
        )


@pytest.mark.parametrize("which", ["P", "A", "PA"])
@pytest.mark.parametrize("encoding", ["indices", "all", "zero-count"])
def test_matrix_update(oracle, available_device, which, encoding):
    fixture = _pinned_update_matrices_fixture()
    data = native_data(fixture)
    c, t = paired_settings(max_iter=1000, scaling=1, eps_abs=1e-5, eps_rel=1e-5)
    workspace = TorchOSQPWorkspace()
    with oracle(**data, **c) as session:
        assert session.setup_code == 0
        original = session.solve()
        x, info = solve(data, available_device, t, workspace)
        assert info["status_val"] == original["status_val"] == 1
        np.testing.assert_allclose(x.cpu().flatten(), original["x"], atol=1e-4, rtol=0)
        args, owners = [], []
        for key, stored in (("P", session._P), ("A", session._A)):
            if key not in which:
                args.extend([None, None, 0])
                continue
            data[key] = fixture[key + "_new"].numpy()
            values = session._values_in_stored_order(data[key], stored)
            indices = np.arange(values.size, dtype=np.int64)
            owners.extend([values, indices])
            args.extend(
                [
                    values.ctypes.data_as(ct.POINTER(OSQPFloat)),
                    indices.ctypes.data_as(ct.POINTER(OSQPInt))
                    if encoding == "indices"
                    else None,
                    0 if encoding == "zero-count" else len(values),
                ]
            )
        assert session.library.osqp_update_data_mat(session._solver, *args) == 0
        reference = session.solve()
        x, info = solve(data, available_device, t, workspace)
        assert info["status_val"] == reference["status_val"] == 1
        np.testing.assert_allclose(x.cpu().flatten(), reference["x"], atol=1e-4, rtol=0)
        np.testing.assert_allclose(
            info["state"]["y"].cpu(), reference["y"], atol=1e-4, rtol=0
        )


@pytest.mark.parametrize("sigma,code", [(1e-6, 4), (5.0, 0)])
def test_nonconvex_setup(oracle, available_device, sigma, code):
    problem = native_data(_pinned_basic_qp2_fixture())
    problem["P"] = np.array([[2.0, 5.0], [5.0, 1.0]])
    c, t = paired_settings(
        sigma=sigma,
        adaptive_rho=0,
        scaling=1,
        max_iter=2000,
        eps_abs=1e-5,
        eps_rel=1e-5,
    )
    with oracle(**problem, **c) as session:
        assert session.setup_code == code
        if code:
            with pytest.raises(TorchLinearSolveError, match="KKT inertia"):
                solve(problem, available_device, t, TorchOSQPWorkspace())
        else:
            reference = session.solve()
            _, info = solve(problem, available_device, t, TorchOSQPWorkspace())
            assert info["status_val"] == reference["status_val"] == 9
