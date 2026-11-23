"""Matrix updates that set a stored entry to zero and back.

``osqp_update_data_mat`` replaces values only: the CSC pattern given to
``osqp_setup`` stays, so an entry that becomes zero is still a stored entry of
the KKT matrix.  The Torch solver must keep the same pattern, ordering and
iterates instead of treating the new zero as a new problem.
"""

import ctypes as ct

import numpy as np
import pytest
import torch
from pinned_osqp_c_oracle import (
    OSQPFloat,
    PinnedOSQPSession,
    paired_settings,
    pinned_dll_path,
)

from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchOSQP import solve_torch_osqp_direct

pytestmark = pytest.mark.skipif(
    not pinned_dll_path().is_file(), reason="pinned OSQP C library is unavailable"
)


def _problem(seed):
    rng = np.random.default_rng(seed)
    n, m = int(rng.integers(4, 10)), int(rng.integers(4, 12))
    upper = np.triu(rng.standard_normal((n, n)) * (rng.random((n, n)) < 0.5), 1)
    P = upper + upper.T
    # Diagonally dominant: P stays positive definite when an off-diagonal is zeroed.
    P += np.diag(np.abs(P).sum(axis=1) + 1.0)
    A = rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.6)
    A[:, 0] += 1.0
    center = A @ rng.standard_normal(n)
    return {
        "P": P,
        "q": rng.standard_normal(n),
        "A": A,
        "l": center - rng.uniform(0.1, 1.0, m),
        "u": center + rng.uniform(0.1, 1.0, m),
    }


def _torch_solve(data, settings, workspace):
    return solve_torch_osqp_direct(
        *(torch.tensor(data[k], dtype=torch.float64) for k in ("P", "q", "A", "l", "u")),
        settings,
        workspace,
    )


def _c_update(session, data, which):
    args, owners = [], []
    for key, stored in (("P", session._P), ("A", session._A)):
        if key != which:
            args.extend([None, None, 0])
            continue
        values = session._values_in_stored_order(data[key], stored)
        owners.append(values)
        args.extend([values.ctypes.data_as(ct.POINTER(OSQPFloat)), None, len(values)])
    return session.library.osqp_update_data_mat(session._solver, *args)


def _assert_same(reference, x, info):
    assert info["status_val"] == reference["status_val"]
    assert info["admm_iterations"] == reference["iter"]
    assert info["rho_updates"] == reference["rho_updates"]
    for actual, expected in (
        (x.flatten(), reference["x"]),
        (info["state"]["y"], reference["y"]),
    ):
        actual = np.asarray(actual, dtype="<f8")
        assert actual.tobytes() == np.asarray(expected, dtype="<f8").tobytes()


@pytest.mark.parametrize("which", ["A", "P"])
@pytest.mark.parametrize("seed", range(20))
def test_stored_entry_set_to_zero_and_restored(seed, which):
    data = _problem(seed)
    original = data[which].copy()
    rng = np.random.default_rng(900 + seed)
    if which == "A":
        candidates = np.argwhere(original != 0)
    else:
        candidates = np.argwhere(np.triu(original, 1) != 0)
    if len(candidates) == 0:
        pytest.skip("no stored off-diagonal entry in this problem")
    row, column = candidates[rng.integers(len(candidates))]
    zeroed = original.copy()
    zeroed[row, column] = 0.0
    if which == "P":
        zeroed[column, row] = 0.0

    c, t = paired_settings()
    workspace = TorchOSQPWorkspace()
    with PinnedOSQPSession(**data, **c) as session:
        assert session.setup_code == 0
        _assert_same(session.solve(), *_torch_solve(data, t, workspace))
        for values in (zeroed, original):
            data[which] = values
            assert _c_update(session, data, which) == 0
            _assert_same(session.solve(), *_torch_solve(data, t, workspace))
