"""Sparse matrix input, compared with pinned C on the same CSC storage.

OSQP takes ``P`` (upper triangle) and ``A`` in CSC form and keeps every stored
entry, explicit zeros included: a stored zero is part of the KKT pattern and
changes the AMD ordering, so it changes the rounded result.  The Torch solver
must use the stored entries of a sparse tensor exactly as given.
"""

import ctypes as ct

import numpy as np
import pinned_osqp_c_oracle as oracle
import pytest
import torch
from pinned_osqp_c_oracle import (
    OSQPFloat,
    PinnedOSQPSession,
    _as_float,
    _as_index,
    paired_settings,
    pinned_dll_path,
)

import pygranso.private.torchCSC as torch_csc
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchOSQP import solve_torch_osqp_direct

pytestmark = pytest.mark.skipif(
    not pinned_dll_path().is_file(), reason="pinned OSQP C library is unavailable"
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _problem(seed):
    rng = np.random.default_rng(seed)
    n, m = int(rng.integers(4, 12)), int(rng.integers(3, 14))
    upper = np.triu(rng.standard_normal((n, n)) * (rng.random((n, n)) < 0.3), 1)
    P = upper + upper.T
    P += np.diag(np.abs(P).sum(axis=1) + 1.0)
    A = rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.4)
    A[:, 0] += 1.0
    center = A @ rng.standard_normal(n)
    data = {
        "P": P,
        "q": rng.standard_normal(n),
        "A": A,
        "l": center - rng.uniform(0.1, 1.0, m),
        "u": center + rng.uniform(0.1, 1.0, m),
    }
    # Stored entries: every nonzero plus some explicit zeros.
    stored = {
        "P": np.triu((P != 0) | (rng.random(P.shape) < 0.3)),
        "A": (A != 0) | (rng.random(A.shape) < 0.3),
    }
    return data, stored


def _csc(matrix, mask):
    """Values, row indices and column pointers of the entries selected by ``mask``."""

    rows, values, pointers = [], [], [0]
    for column in range(matrix.shape[1]):
        index = np.flatnonzero(mask[:, column])
        rows.extend(index.tolist())
        values.extend(matrix[index, column].tolist())
        pointers.append(len(rows))
    return _as_float(values), _as_index(rows), _as_index(pointers)


class _StoredSession(PinnedOSQPSession):
    """A C session whose CSC inputs are the given stored entries."""

    def __init__(self, data, stored, monkeypatch, **settings):
        def from_mask(matrix, *, upper_triangular=False):
            return _csc(np.asarray(matrix), stored["P" if upper_triangular else "A"])

        monkeypatch.setattr(oracle, "dense_to_csc", from_mask)
        super().__init__(**data, **settings)

    def update_values(self, data, which):
        args, owners = [], []
        for key, mask in (("P", self._P), ("A", self._A)):
            if key != which:
                args.extend([None, None, 0])
                continue
            values = self._values_in_stored_order(data[key], mask)
            owners.append(values)
            args.extend([values.ctypes.data_as(ct.POINTER(OSQPFloat)), None, len(values)])
        return self.library.osqp_update_data_mat(self._solver, *args)


def _sparse(matrix, mask, layout, device):
    values, rows, pointers = _csc(matrix, mask)
    tensor = torch.sparse_csc_tensor(
        torch.from_numpy(pointers.astype(np.int64)),
        torch.from_numpy(rows.astype(np.int64)),
        torch.from_numpy(values.astype(np.float64)),
        size=matrix.shape,
        dtype=torch.float64,
    )
    if layout == "csr":
        columns = np.repeat(np.arange(matrix.shape[1]), np.diff(pointers))
        order = np.lexsort((columns, rows))
        counts = np.bincount(rows, minlength=matrix.shape[0])
        tensor = torch.sparse_csr_tensor(
            torch.from_numpy(np.concatenate(([0], np.cumsum(counts))).astype(np.int64)),
            torch.from_numpy(columns[order].astype(np.int64)),
            torch.from_numpy(values[order].astype(np.float64)),
            size=matrix.shape,
            dtype=torch.float64,
        )
    elif layout == "coo":
        columns = np.repeat(np.arange(matrix.shape[1]), np.diff(pointers))
        tensor = torch.sparse_coo_tensor(
            torch.from_numpy(np.stack((rows, columns)).astype(np.int64)),
            torch.from_numpy(values.astype(np.float64)),
            size=matrix.shape,
            dtype=torch.float64,
        )
    return tensor.to(device)


def _torch_solve(data, stored, layout, device, settings, workspace):
    def vector(key):
        return torch.tensor(data[key], dtype=torch.float64, device=device)

    settings = dict(settings, sparse=True)

    return solve_torch_osqp_direct(
        _sparse(data["P"], stored["P"], layout, device),
        vector("q"),
        _sparse(data["A"], stored["A"], layout, device),
        vector("l"),
        vector("u"),
        settings,
        workspace,
    )


def _assert_same(reference, x, info):
    assert info["status_val"] == reference["status_val"]
    assert info["admm_iterations"] == reference["iter"]
    assert info["rho_updates"] == reference["rho_updates"]
    # store_solution writes x and y for every status (NaN without a solution).
    for actual, expected in (
        (x.flatten(), reference["x"]),
        (info["public_y"], reference["y"]),
    ):
        if actual.is_cuda:
            np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-7, rtol=1e-7)
            continue
        actual = np.asarray(actual.cpu(), dtype="<f8")
        assert actual.tobytes() == np.asarray(expected, dtype="<f8").tobytes()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layout", ["csc", "csr", "coo"])
@pytest.mark.parametrize("seed", range(12))
def test_stored_zeros_of_sparse_input_are_kept(seed, layout, device, monkeypatch):
    data, stored = _problem(seed)
    c, t = paired_settings(polishing=1)
    with _StoredSession(data, stored, monkeypatch, **c) as session:
        assert session.setup_code == 0
        reference = session.solve()
    x, info = _torch_solve(data, stored, layout, device, t, TorchOSQPWorkspace())
    _assert_same(reference, x, info)


@pytest.mark.parametrize("seed", range(12))
def test_stored_zeros_change_the_native_result(seed, monkeypatch):
    """The comparison above is not vacuous: C itself depends on the stored zeros."""

    data, stored = _problem(seed)
    nonzeros = {"P": np.triu(data["P"] != 0), "A": data["A"] != 0}
    if all((stored[key] == nonzeros[key]).all() for key in stored):
        pytest.skip("no explicit zero in this problem")
    c, _ = paired_settings()
    with _StoredSession(data, stored, monkeypatch, **c) as session:
        with_zeros = session.solve()
    with _StoredSession(data, nonzeros, monkeypatch, **c) as session:
        without = session.solve()
    assert with_zeros["status_val"] == without["status_val"] == 1
    assert (
        with_zeros["x"].tobytes() != without["x"].tobytes()
        or with_zeros["y"].tobytes() != without["y"].tobytes()
        or with_zeros["iter"] != without["iter"]
    )


@pytest.mark.parametrize("which", ["P", "A"])
@pytest.mark.parametrize("seed", range(12))
def test_sparse_value_update_keeps_the_pattern(seed, which, monkeypatch):
    data, stored = _problem(seed)
    rng = np.random.default_rng(700 + seed)
    updated = dict(data)
    if which == "A":
        updated["A"] = data["A"] * rng.uniform(0.5, 1.5, data["A"].shape)
        position = np.argwhere(data["A"] != 0)[0]
        updated["A"][tuple(position)] = 0.0
    else:
        factor = np.triu(rng.uniform(0.9, 1.1, data["P"].shape), 1)
        updated["P"] = data["P"] * (factor + factor.T + np.eye(len(data["q"])))
    c, t = paired_settings()
    workspace = TorchOSQPWorkspace()
    with _StoredSession(data, stored, monkeypatch, **c) as session:
        assert session.setup_code == 0
        _assert_same(session.solve(), *_torch_solve(data, stored, "csc", "cpu", t, workspace))
        pattern = workspace.a_pattern
        assert session.update_values(updated, which) == 0
        _assert_same(
            session.solve(), *_torch_solve(updated, stored, "csc", "cpu", t, workspace)
        )
    assert workspace.a_pattern is pattern


@pytest.mark.parametrize("seed", range(12))
def test_strided_and_sparse_inputs_agree(seed):
    data, _ = _problem(seed)
    nonzeros = {"P": np.triu(data["P"] != 0), "A": data["A"] != 0}
    _, t = paired_settings(polishing=1)
    dense = [torch.tensor(data[key], dtype=torch.float64) for key in ("P", "q", "A", "l", "u")]
    x_dense, info_dense = solve_torch_osqp_direct(*dense, t)
    x_sparse, info_sparse = _torch_solve(data, nonzeros, "csc", "cpu", t, TorchOSQPWorkspace())
    assert info_dense["admm_iterations"] == info_sparse["admm_iterations"]
    assert x_dense.numpy().tobytes() == x_sparse.numpy().tobytes()
    assert info_dense["state"]["y"].numpy().tobytes() == info_sparse["state"]["y"].numpy().tobytes()


@pytest.mark.parametrize("seed", range(12))
def test_per_rank_accumulation_matches_native(seed, monkeypatch):
    """The accumulation used off CPU float64, forced on CPU, against C."""

    data, stored = _problem(seed)
    c, t = paired_settings(polishing=1)
    with _StoredSession(data, stored, monkeypatch, **c) as session:
        reference = session.solve()
    monkeypatch.setattr(torch_csc, "FORCE_LEVEL_LOOP", True)
    x, info = _torch_solve(data, stored, "csc", "cpu", t, TorchOSQPWorkspace())
    _assert_same(reference, x, info)


@pytest.mark.parametrize("seed", range(12))
def test_without_the_sparse_setting_stored_zeros_are_dropped(seed, monkeypatch):
    """The default reads a sparse matrix as the strided matrix with the same values."""

    data, stored = _problem(seed)
    nonzeros = {"P": np.triu(data["P"] != 0), "A": data["A"] != 0}
    c, t = paired_settings()
    with _StoredSession(data, nonzeros, monkeypatch, **c) as session:
        reference = session.solve()
    x, info = solve_torch_osqp_direct(
        _sparse(data["P"], stored["P"], "csc", "cpu"),
        torch.tensor(data["q"], dtype=torch.float64),
        _sparse(data["A"], stored["A"], "csc", "cpu"),
        torch.tensor(data["l"], dtype=torch.float64),
        torch.tensor(data["u"], dtype=torch.float64),
        t,
    )
    _assert_same(reference, x, info)
