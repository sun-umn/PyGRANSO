"""Exact parity of the Python AMD / KKT-assembly ports with the pinned C OSQP.

For every case the C library is set up with ``scaling=0`` and its linear-system
workspace is read back.  The Python ports (``form_KKT`` -> ``amd_order`` ->
``csc_pinv`` -> ``csc_symperm``) must then reproduce

* the AMD permutation exactly,
* the permuted KKT pattern (``p``, ``i``) exactly,
* the permuted KKT values bit for bit,
* the permuted index maps ``PtoKKT`` / ``AtoKKT`` / ``rhotoKKT`` exactly.

Run ``python tests/test_c_amd_kkt_parity.py`` for a one-line-per-family sweep
summary with the exact number of cases compared and matched.
"""

import ctypes as ct
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parent
for _path in (str(_TESTS), str(_TESTS.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pinned_osqp_c_internals import Workspace, read_workspace  # noqa: E402
from pinned_osqp_c_oracle import dense_to_csc, pinned_dll_path  # noqa: E402
from pinned_osqp_c_session import PinnedOSQPSession, paired_settings  # noqa: E402

from pygranso.private.torchAMD import (  # noqa: E402
    AMD_INVALID,
    AMD_OK,
    AMD_OK_BUT_JUMBLED,
    amd_order,
)
from pygranso.private.torchKKTForm import (  # noqa: E402
    csc_pinv,
    csc_symperm,
    form_KKT,
    permute_KKT,
)

pytestmark = pytest.mark.skipif(
    not pinned_dll_path().is_file(), reason="pinned OSQP C library is unavailable"
)

INF = 1e30


# --------------------------------------------------------------------------- #
# problem generators (all deterministic)
# --------------------------------------------------------------------------- #

def _psd_on_pattern(rng, n, density, zero_fraction=0.0):
    """Symmetric diagonally dominant matrix (hence PSD) on a random pattern.

    Rows/columns selected by ``zero_fraction`` are entirely zero, which gives
    missing diagonal entries in the CSC form of P.
    """
    M = rng.standard_normal((n, n))
    S = M @ M.T
    mask = np.triu(rng.random((n, n)) < density, 1)
    mask = mask | mask.T
    P = np.where(mask, S, 0.0)
    dead = rng.random(n) < zero_fraction
    P[dead, :] = 0.0
    P[:, dead] = 0.0
    diag = np.abs(P).sum(axis=1) + rng.uniform(0.1, 2.0, n)
    diag[dead] = 0.0
    P[np.arange(n), np.arange(n)] = diag
    return P


def _random_P(rng, n, kind):
    if kind == "zero":
        return np.zeros((n, n))
    if kind == "diag":
        return np.diag(rng.uniform(0.1, 3.0, n))
    if kind == "diag_missing":
        d = rng.uniform(0.1, 3.0, n)
        d[rng.random(n) < 0.5] = 0.0
        return np.diag(d)
    if kind == "dense":
        M = rng.standard_normal((n, n))
        return M @ M.T + 1e-3 * np.eye(n)
    if kind == "sparse":
        return _psd_on_pattern(rng, n, rng.choice([0.05, 0.15, 0.4]))
    if kind == "very_sparse":
        return _psd_on_pattern(rng, n, 1.5 / max(n, 2))
    if kind == "sparse_missing":
        return _psd_on_pattern(rng, n, rng.choice([0.1, 0.3]), zero_fraction=0.35)
    raise ValueError(kind)


def _random_A(rng, m, n, kind):
    if m == 0:
        return np.zeros((0, n))
    if kind == "dense":
        A = rng.standard_normal((m, n))
    elif kind == "sparse":
        A = rng.standard_normal((m, n)) * (rng.random((m, n)) < rng.choice([0.1, 0.3]))
    elif kind == "very_sparse":
        A = rng.standard_normal((m, n)) * (rng.random((m, n)) < 1.2 / max(n, 2))
    elif kind == "empty_rows_cols":
        A = rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.3)
        A[rng.random(m) < 0.3, :] = 0.0
        A[:, rng.random(n) < 0.3] = 0.0
    elif kind == "duplicate_rows":
        A = rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.35)
        for _ in range(max(1, m // 3)):
            src, dst = rng.integers(0, m, 2)
            A[dst] = A[src]
    elif kind == "zero":
        A = np.zeros((m, n))
    else:
        raise ValueError(kind)
    return A


def _random_bounds(rng, m):
    l = -rng.uniform(0.5, 2.0, m)
    u = rng.uniform(0.5, 2.0, m)
    kind = rng.integers(0, 5, m)
    eq = kind == 0
    value = rng.standard_normal(m)
    l[eq] = value[eq]
    u[eq] = value[eq]
    l[kind == 1] = -INF
    u[kind == 2] = INF
    free = kind == 3
    if rng.random() < 0.2:
        l[free] = -INF
        u[free] = INF
    return l, u


_P_KINDS = ("zero", "diag", "diag_missing", "dense", "sparse", "very_sparse",
            "sparse_missing")
_A_KINDS = ("dense", "sparse", "very_sparse", "empty_rows_cols",
            "duplicate_rows", "zero")


def _random_case(index):
    rng = np.random.default_rng(900000 + index)
    p_kind = _P_KINDS[index % len(_P_KINDS)]
    a_kind = _A_KINDS[(index // len(_P_KINDS)) % len(_A_KINDS)]
    # sizes: sweep n over 1..40 exactly, m over 0..60 with m = 0 regularly
    n = 1 + (index * 7) % 40
    m = 0 if index % 11 == 0 else int(rng.integers(0, 61))
    P = _random_P(rng, n, p_kind)
    A = _random_A(rng, m, n, a_kind)
    l, u = _random_bounds(rng, m)
    q = rng.standard_normal(n)
    return P, q, A, l, u


def _banded_case(index):
    rng = np.random.default_rng(910000 + index)
    n = 5 + 3 * index
    m = (n + index) % 61
    bw_p = 1 + index % 4
    bw_a = 1 + index % 3
    P = np.zeros((n, n))
    for k in range(1, bw_p + 1):
        v = rng.standard_normal(n - k) if n > k else np.zeros(0)
        P += np.diag(v, k) + np.diag(v, -k)
    P[np.arange(n), np.arange(n)] = np.abs(P).sum(axis=1) + 0.5
    A = np.zeros((m, n))
    for r in range(m):
        c0 = (r * n) // max(m, 1)
        cols = np.arange(max(0, c0 - bw_a), min(n, c0 + bw_a + 1))
        A[r, cols] = rng.standard_normal(cols.size)
    l, u = _random_bounds(rng, m)
    return P, rng.standard_normal(n), A, l, u


def _arrow_small_case(index):
    """Arrow / star patterns below the dense-row threshold."""
    rng = np.random.default_rng(920000 + index)
    n = 4 + 3 * index
    m = (5 * index) % 40
    P = np.diag(rng.uniform(1.0, 2.0, n))
    hub = index % n
    v = rng.uniform(-1.0, 1.0, n) / n
    P[hub, :] = v
    P[:, hub] = v
    P[hub, hub] = 2.0
    A = rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.08)
    if m:
        A[index % m, :] = rng.standard_normal(n)      # one full row
        A[:, (index * 3) % n] = rng.standard_normal(m)  # one full column
    l, u = _random_bounds(rng, m)
    return P, rng.standard_normal(n), A, l, u


def _dense_row_case(index):
    """KKT systems with rows denser than max(16, 10 sqrt(n+m)): AMD's dense path."""
    rng = np.random.default_rng(930000 + index)
    variant = index % 6
    if variant == 0:      # one full constraint row
        n, m, rows, cols, hubs = 200, 150, 1, 0, 0
    elif variant == 1:    # one variable in every constraint
        n, m, rows, cols, hubs = 150, 250, 0, 1, 0
    elif variant == 2:    # arrow P with a dense hub variable, few constraints
        n, m, rows, cols, hubs = 330, 20, 0, 0, 1
    elif variant == 3:    # several dense rows and columns together
        n, m, rows, cols, hubs = 220, 200, 3, 2, 1
    elif variant == 4:    # dense rows only, no P coupling, m = small
        n, m, rows, cols, hubs = 300, 12, 4, 0, 0
    else:                 # two hubs in P plus dense rows; duplicate dense rows
        n, m, rows, cols, hubs = 260, 120, 2, 1, 2
    n += index // 6
    P = np.diag(rng.uniform(1.0, 2.0, n))
    if index % 2:
        k = 1 + index % 3
        v = rng.uniform(-0.2, 0.2, n - k)
        P += np.diag(v, k) + np.diag(v, -k)
    for h in range(hubs):
        hub = int(rng.integers(0, n))
        v = rng.uniform(-1.0, 1.0, n) / (4.0 * n)
        v[hub] = 0.0
        P[hub, :] += v
        P[:, hub] += v
    P[np.arange(n), np.arange(n)] = 0.0
    P[np.arange(n), np.arange(n)] = np.abs(P).sum(axis=1) + 1.0
    A = rng.standard_normal((m, n)) * (rng.random((m, n)) < 1.5 / n)
    row_ids = rng.choice(m, size=rows, replace=False) if rows else []
    for r in row_ids:
        A[r, :] = rng.standard_normal(n)
    if variant == 5 and rows >= 2:
        A[row_ids[1], :] = A[row_ids[0], :]
    for c in (rng.choice(n, size=cols, replace=False) if cols else []):
        A[:, c] = rng.standard_normal(m)
    l, u = _random_bounds(rng, m)
    return P, rng.standard_normal(n), A, l, u


def _fill_heavy_case(index):
    """Larger random sparse systems: enough fill to force AMD garbage collection."""
    rng = np.random.default_rng(940000 + index)
    n = 90 + 10 * index
    m = 60 + 15 * index
    P = _psd_on_pattern(rng, n, 2.0 / n)
    A = rng.standard_normal((m, n)) * (rng.random((m, n)) < 2.5 / n)
    l, u = _random_bounds(rng, m)
    return P, rng.standard_normal(n), A, l, u


def _edge_case(index):
    rng = np.random.default_rng(950000 + index)
    if index == 0:      # 1 x 1, no constraints, P = 0
        return np.zeros((1, 1)), np.ones(1), np.zeros((0, 1)), np.zeros(0), np.zeros(0)
    if index == 1:      # 1 x 1 with one constraint
        return np.array([[2.0]]), np.ones(1), np.array([[1.0]]), -np.ones(1), np.ones(1)
    if index == 2:      # P = 0, A = 0
        return np.zeros((6, 6)), np.zeros(6), np.zeros((4, 6)), -np.ones(4), np.ones(4)
    if index == 3:      # identity everything
        return np.eye(7), np.zeros(7), np.eye(7), -np.ones(7), np.ones(7)
    if index == 4:      # all rows identical
        A = np.tile(rng.standard_normal(9), (12, 1))
        return np.eye(9), np.zeros(9), A, -np.ones(12), np.ones(12)
    if index == 5:      # fully dense P and A at the largest sweep size
        M = rng.standard_normal((40, 40))
        A = rng.standard_normal((60, 40))
        return M @ M.T + np.eye(40), np.zeros(40), A, -np.ones(60), np.ones(60)
    if index == 6:      # all equality rows
        A = rng.standard_normal((10, 20)) * (rng.random((10, 20)) < 0.3)
        b = rng.standard_normal(10)
        return _psd_on_pattern(rng, 20, 0.2), np.zeros(20), A, b, b.copy()
    if index == 7:      # all one-sided / free rows
        A = rng.standard_normal((15, 12)) * (rng.random((15, 12)) < 0.3)
        l = np.full(15, -INF)
        u = np.full(15, INF)
        u[:5] = 1.0
        return _psd_on_pattern(rng, 12, 0.2), np.zeros(12), A, l, u
    if index == 8:      # block diagonal: disconnected components
        P = np.zeros((30, 30))
        for b in range(0, 30, 5):
            M = rng.standard_normal((5, 5))
            P[b:b + 5, b:b + 5] = M @ M.T + np.eye(5)
        A = np.zeros((6, 30))
        for r in range(6):
            A[r, 5 * r:5 * r + 5] = rng.standard_normal(5)
        return P, np.zeros(30), A, -np.ones(6), np.ones(6)
    if index == 9:      # m much larger than n, single column
        A = rng.standard_normal((60, 1))
        return np.array([[1.0]]), np.ones(1), A, -np.ones(60), np.ones(60)
    raise IndexError(index)


FAMILIES = (
    ("random", _random_case, 336),
    ("banded", _banded_case, 20),
    ("arrow_small", _arrow_small_case, 20),
    ("dense_row", _dense_row_case, 18),
    ("fill_heavy", _fill_heavy_case, 10),
    ("edge", _edge_case, 10),
)
CASES = [(name, index) for name, _, count in FAMILIES for index in range(count)]
_BUILDERS = {name: builder for name, builder, _ in FAMILIES}


# --------------------------------------------------------------------------- #
# comparison against C
# --------------------------------------------------------------------------- #

def _csc_dict(triple, rows, cols):
    x, i, p = triple
    return {"m": rows, "n": cols, "p": p.tolist(), "i": i.tolist(), "x": x.tolist()}


def _c_index_maps(session, nnzP, nnzA, m):
    work = ct.cast(session._solver.contents.work, ct.POINTER(Workspace)).contents
    lin = work.linsys_solver.contents

    def read(pointer, size):
        return [int(pointer[k]) for k in range(size)]

    return read(lin.PtoKKT, nnzP), read(lin.AtoKKT, nnzA), read(lin.rhotoKKT, m)


def _bits_equal(py_values, c_values):
    a = np.asarray(py_values, dtype=np.float64)
    b = np.asarray(c_values, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a, b) and a.tobytes() == b.tobytes()


def compare_case(P, q, A, l, u):
    """Return ``None`` if C setup fails, else a dict of exact-match booleans."""
    n = int(np.asarray(q).size)
    m = int(np.asarray(l).size)
    A = np.asarray(A, dtype=np.float64).reshape(m, n)
    c_settings, _ = paired_settings(scaling=0)
    with PinnedOSQPSession(P, q, A, l, u, **c_settings) as session:
        if session.setup_code != 0:
            return None
        w = read_workspace(session._solver)
        P_csc = _csc_dict(dense_to_csc(np.asarray(P, dtype=np.float64),
                                       upper_triangular=True), n, n)
        A_csc = _csc_dict(dense_to_csc(A), m, n)
        c_maps = _c_index_maps(session, P_csc["p"][n], A_csc["p"][n], m)

    lin = w["linsys"]
    # scaling=0: the C work data must be the input data
    inputs_equal = all(
        mine["p"] == theirs["p"].tolist() and mine["i"] == theirs["i"].tolist()
        and _bits_equal(mine["x"], theirs["x"])
        for mine, theirs in ((P_csc, w["P"]), (A_csc, w["A"]))
    )

    rho_inv_vec = w["rho_inv_vec"].tolist()
    assert len(rho_inv_vec) == m
    KKT, PtoKKT, AtoKKT, rhotoKKT = form_KKT(
        P_csc, A_csc, 0, lin["sigma"], rho_inv_vec, 1.0 / c_settings["rho"],
        return_maps=True,
    )
    stats = {}
    status, perm = amd_order(KKT["n"], KKT["p"], KKT["i"], stats=stats)
    pinv = csc_pinv(perm)
    KtoPKPt = [0] * KKT["p"][KKT["n"]]
    KKTp = csc_symperm(KKT, pinv, KtoPKPt, 1)
    maps = tuple([KtoPKPt[k] for k in mapping] for mapping in (PtoKKT, AtoKKT, rhotoKKT))

    # the composed permute_KKT port must agree with the manual pipeline
    status2, perm2, KKTp2 = permute_KKT(KKT, PtoKKT, AtoKKT, rhotoKKT)
    composed = (
        status2 == status and perm2 == perm and KKTp2["p"] == KKTp["p"]
        and KKTp2["i"] == KKTp["i"] and _bits_equal(KKTp2["x"], KKTp["x"])
        and (PtoKKT, AtoKKT, rhotoKKT) == maps
    )

    dense_threshold = min(n + m, max(16, int(10.0 * math.sqrt(float(n + m)))))
    return {
        "status": status == AMD_OK,
        "inputs": inputs_equal,
        "perm": perm == lin["perm"].tolist(),
        "pattern": (KKTp["p"] == lin["KKT"]["p"].tolist()
                    and KKTp["i"] == lin["KKT"]["i"].tolist()),
        "values": _bits_equal(KKTp["x"], lin["KKT"]["x"]),
        "maps": maps == c_maps,
        "composed": composed,
        "ndense": stats["ndense"],
        "ncmpa": stats["ncmpa"],
        "size": n + m,
        "dense_threshold": dense_threshold,
    }


_CHECKS = ("status", "inputs", "perm", "pattern", "values", "maps", "composed")


@pytest.mark.parametrize("family,index", CASES, ids=[f"{f}-{i}" for f, i in CASES])
def test_amd_and_permuted_kkt_match_c(family, index):
    result = compare_case(*_BUILDERS[family](index))
    if result is None:
        pytest.skip("C setup returned a nonzero code")
    failed = [name for name in _CHECKS if not result[name]]
    assert not failed, f"{family}-{index}: mismatch in {failed}"


def run_sweep(cases=CASES):
    summary = {}
    for family, index in cases:
        row = summary.setdefault(
            family, {"cases": 0, "skipped": 0, "compared": 0, "dense": 0,
                     "compressed": 0, "max_size": 0, "failures": [],
                     **{name: 0 for name in _CHECKS}},
        )
        row["cases"] += 1
        result = compare_case(*_BUILDERS[family](index))
        if result is None:
            row["skipped"] += 1
            continue
        row["compared"] += 1
        row["dense"] += int(result["ndense"] > 0)
        row["compressed"] += int(result["ncmpa"] > 0)
        row["max_size"] = max(row["max_size"], result["size"])
        for name in _CHECKS:
            row[name] += int(result[name])
        if not all(result[name] for name in _CHECKS):
            row["failures"].append(index)
    return summary


def test_sweep_covers_required_regimes():
    """The dense-row path and garbage collection are actually exercised, and
    every case of those families matches C exactly."""
    subset = [case for case in CASES if case[0] in ("dense_row", "fill_heavy")]
    summary = run_sweep(subset)
    dense = summary["dense_row"]
    heavy = summary["fill_heavy"]
    assert dense["skipped"] == 0 and heavy["skipped"] == 0
    assert dense["dense"] == dense["compared"] == dense["cases"]
    assert dense["max_size"] > 300
    assert heavy["compressed"] >= 1
    for row in (dense, heavy):
        assert not row["failures"]


def test_random_sweep_size_coverage():
    sizes = {(1 + (index * 7) % 40) for index in range(336)}
    assert sizes == set(range(1, 41))
    assert len(CASES) >= 300


# --------------------------------------------------------------------------- #
# pure-Python checks of branches the C oracle cannot reach
# (the KKT handed to AMD by OSQP is always sorted and duplicate free)
# --------------------------------------------------------------------------- #

def test_amd_order_trivial_and_invalid_inputs():
    assert amd_order(0, [0], []) == (AMD_OK, [])
    assert amd_order(1, [0, 0], []) == (AMD_OK, [0])
    assert amd_order(-1, [0], []) == (AMD_INVALID, None)
    assert amd_order(2, [1, 1, 1], [0]) == (AMD_INVALID, None)     # Ap[0] != 0
    assert amd_order(2, [0, 2, 1], [0, 1]) == (AMD_INVALID, None)  # descending Ap
    assert amd_order(2, [0, 1, 2], [0, 2]) == (AMD_INVALID, None)  # row out of range


def test_amd_order_jumbled_input_equals_sorted_transpose_pattern():
    """AMD_preprocess path: a jumbled matrix is ordered as R = A' (sorted, no
    duplicates), so the result must equal ordering that matrix directly."""
    rng = np.random.default_rng(7)
    for n in (5, 17, 40):
        dense = rng.random((n, n)) < 0.2
        Ap, Ai = [0], []
        for j in range(n):
            rows = np.flatnonzero(dense[:, j]).tolist()
            rows = rows[::-1] + rows[:1]          # unsorted, with a duplicate
            Ai.extend(rows)
            Ap.append(len(Ai))
        Tp, Ti = [0], []
        for j in range(n):                        # sorted pattern of A'
            Ti.extend(np.flatnonzero(dense[j, :]).tolist())
            Tp.append(len(Ti))
        stats = {}
        status, perm = amd_order(n, Ap, Ai, stats=stats)
        assert stats["jumbled"] == 1
        assert status == AMD_OK_BUT_JUMBLED
        assert sorted(perm) == list(range(n))
        assert (AMD_OK, perm) == amd_order(n, Tp, Ti)


def test_form_kkt_scalar_param2_and_maps():
    P = {"m": 3, "n": 3, "p": [0, 1, 2, 2], "i": [0, 0], "x": [4.0, 1.0]}
    A = {"m": 2, "n": 3, "p": [0, 1, 1, 3], "i": [1, 0, 1], "x": [5.0, 6.0, 7.0]}
    KKT, PtoKKT, AtoKKT, rhotoKKT = form_KKT(P, A, 0, 0.5, None, 0.25, return_maps=True)
    assert KKT["p"] == [0, 1, 3, 4, 6, 9]
    assert KKT["i"] == [0, 0, 1, 2, 2, 3, 0, 2, 4]
    assert KKT["x"] == [4.5, 1.0, 0.5, 0.5, 6.0, -0.25, 5.0, 7.0, -0.25]
    assert (PtoKKT, AtoKKT, rhotoKKT) == ([0, 1], [6, 4, 7], [5, 8])
    vector = form_KKT(P, A, 0, 0.5, [0.25, 0.25], 99.0)
    assert vector["p"] == KKT["p"] and vector["i"] == KKT["i"] and vector["x"] == KKT["x"]
    # identity permutation: symperm keeps an already upper-triangular matrix
    same = csc_symperm(KKT, csc_pinv(list(range(5))), None, 1)
    assert same["p"] == KKT["p"] and same["i"] == KKT["i"] and same["x"] == KKT["x"]
    assert csc_symperm(KKT, None, None, 0)["x"] is None


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    totals = {"cases": 0, "skipped": 0, "compared": 0,
              **{name: 0 for name in _CHECKS}}
    for family, row in run_sweep().items():
        print(family, {k: v for k, v in row.items()})
        for key in totals:
            totals[key] += row[key]
    print("TOTAL", totals, f"elapsed {time.perf_counter() - start:.1f}s")
