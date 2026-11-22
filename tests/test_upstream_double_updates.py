"""Observable counterparts of the pinned basic_qp and update_matrices tests.

The standalone native runner covers C-only APIs. Here all three C index
encodings are exercised against the same tensor matrix submission.
"""

import ctypes as ct
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pinned_osqp_c_oracle import (
    OSQPFloat,
    OSQPInt,
    PinnedOSQPSession,
    paired_settings,
    pinned_dll_path,
)

from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchLinearSolve import TorchLinearSolveError
from pygranso.private.torchOSQP import solve_torch_osqp_direct

CASES = json.loads(Path(__file__).with_name("pinned_double_results.json").read_text())


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def available_device(request):
    return torch.device(request.param)


def problem(name):
    return {
        key: np.asarray(value, dtype=np.float64)
        for key, value in zip(("P", "q", "A", "l", "u"), CASES[name]["qp"])
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


@pytest.mark.parametrize("change", ["q", "l", "u", "flipped", "bad-l", "bad-u", "both"])
def test_vector_update(oracle, available_device, change):
    data = problem("basic_qp")
    new = {
        "q": np.array([2.5, 3.2]),
        "l": np.array([0.8, -3.4, -1e30, 0.5]),
        "u": np.array([1.6, 1.0, 1e30, 0.5]),
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
        installed = session.problem_vectors()
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
    data = problem("update_matrices")
    changes = CASES["update_matrices"]["updated_matrices"]
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
            data[key] = np.asarray(changes[key], dtype=np.float64)
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
    data = problem("basic_qp2")
    data["P"] = np.array([[2.0, 5.0], [5.0, 1.0]])
    c, t = paired_settings(
        sigma=sigma,
        adaptive_rho=0,
        scaling=1,
        max_iter=2000,
        eps_abs=1e-5,
        eps_rel=1e-5,
    )
    with oracle(**data, **c) as session:
        assert session.setup_code == code
        if code:
            with pytest.raises(TorchLinearSolveError, match="KKT inertia"):
                solve(data, available_device, t, TorchOSQPWorkspace())
        else:
            reference = session.solve()
            _, info = solve(data, available_device, t, TorchOSQPWorkspace())
            assert info["status_val"] == reference["status_val"] == 9
