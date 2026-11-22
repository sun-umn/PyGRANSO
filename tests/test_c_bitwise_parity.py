"""Bit-for-bit agreement between the Torch translation and the pinned OSQP C DLL.

Numerical comparisons include all binary64 bits.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from c_parity_support import (
    PinnedOSQPSession,
    TorchOSQPWorkspace,
    compare_factors,
    compare_setup,
    compare_solve,
    paired_settings,
    random_qp,
    read_workspace,
    same_bits,
    torch_solve,
)
from pinned_osqp_c_oracle import pinned_dll_path

from pygranso.private import torchOSQP
from pygranso.private.torchLinearSolve import TorchLinearSolveError
from pygranso.private.torchQDLDL import qdldl_etree, qdldl_factor

pytestmark = pytest.mark.skipif(
    not pinned_dll_path().is_file(), reason="pinned OSQP C library is unavailable"
)


@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("sigma", [1e-6, 1.0])
def test_source_setup_failure_and_adjacent_success(rows, sigma):
    problem = (np.array([[-sigma]]), np.zeros(1), np.ones((rows, 1)),
               -np.ones(rows), np.ones(rows))
    c, t = paired_settings(sigma=sigma, scaling=0, polishing=0, adaptive_rho=0)
    workspace = TorchOSQPWorkspace()
    with PinnedOSQPSession(*problem, **c) as session:
        if rows == 1:
            assert session.setup_code == 4
            with pytest.raises(TorchLinearSolveError, match="zeros in the diagonal"):
                torch_solve(problem, t, workspace)
            assert workspace.last_info is None
        else:
            assert session.setup_code == 0
            result = session.solve()
            x, info = torch_solve(problem, t, workspace)
            assert result["status_val"] == info["status_val"] == 1
            assert compare_solve(result, read_workspace(session._solver), x, info, workspace) == []


def test_solveqp_retains_failure_information_and_reuses_successful_workspace():
    from pygranso.private.solveQP import getLastOSQPInfo, solveQP

    ws = TorchOSQPWorkspace()
    p = torch.eye(1, dtype=torch.float64)
    q = p.new_zeros((1, 1))
    opts = {"workspace": ws, "settings": {"polishing": True, "return_info": True}}

    def run(lower):
        return solveQP(p, q, p, p.new_zeros(1), p.new_full((1, 1), lower),
                       p.new_ones((1, 1)), "osqp", p.device, True, opts)

    x, first = run(-1.)
    assert first["status_val"] == 1 and x.item() == 0.
    factors = ws.linear_solver.matrix
    committed = ws.work_iterates
    with pytest.raises(RuntimeError, match="primal_infeasible"):
        run(0.5)
    failure = getLastOSQPInfo(ws)
    assert failure["status_val"] == 3 and failure["objective"] == 1e30
    # The incoming x is exactly zero. Polishing proposes no improvement.
    assert failure["status_polish"] == first["status_polish"] == -1
    assert ws.work_iterates is committed and ws.linear_solver.matrix is factors
    _, retry = run(-1.)
    assert retry["status_val"] == 1 and ws.linear_solver.matrix is factors

VARIANTS = [
    {},
    {"eps_abs": 1e-8, "eps_rel": 1e-8},
    {"polishing": 1},
    {"polishing": 1, "eps_abs": 1e-6, "eps_rel": 1e-6, "scaling": 0},
    {"scaling": 0},
    {"scaling": 1},
    {"scaling": 3, "alpha": 1.0},
    {"rho": 1.0, "sigma": 1e-3, "alpha": 1.8},
    {"adaptive_rho": 0},
    {"adaptive_rho": 3, "adaptive_rho_interval": 0},
    {"adaptive_rho": 3, "adaptive_rho_interval": 7, "adaptive_rho_fraction": 0.9},
    {"adaptive_rho": 1, "adaptive_rho_interval": 0},
    {"adaptive_rho": 1, "adaptive_rho_interval": 13, "adaptive_rho_tolerance": 1.5},
    {"check_termination": 1},
    {"check_termination": 0, "max_iter": 60},
    {"check_termination": 7, "max_iter": 33},
    {"scaled_termination": 1},
    {"check_dualgap": 0},
    {"max_iter": 1},
    {"max_iter": 2, "check_termination": 1},
    {"max_iter": 10, "eps_abs": 1e-12, "eps_rel": 1e-12},
    {"warm_starting": 0},
]


def _one_solve(problem, overrides):
    c_settings, torch_settings = paired_settings(**overrides)
    workspace = TorchOSQPWorkspace()
    with PinnedOSQPSession(*problem, **c_settings) as session:
        assert session.setup_code == 0
        after_setup = read_workspace(session._solver)
        c_result = session.solve()
        after_solve = read_workspace(session._solver)
    x, info = torch_solve(problem, torch_settings, workspace)
    bad = compare_solve(c_result, after_solve, x, info, workspace)
    bad += compare_setup(after_setup, workspace, torch_settings, problem)
    if c_result["rho_updates"] == 0 and c_result["status_val"] not in (3, 4, 5, 6, 9):
        bad += compare_factors(after_setup, workspace)
    return c_result, bad


@pytest.mark.parametrize("variant", range(len(VARIANTS)))
@pytest.mark.parametrize("seed", range(6))
def test_feasible_solve_is_bit_identical(seed, variant):
    _, bad = _one_solve(random_qp(seed), VARIANTS[variant])
    assert bad == []


@pytest.mark.parametrize("variant", [0, 4, 8, 12, 16, 20])
@pytest.mark.parametrize("kind", ["infeasible", "unbounded"])
@pytest.mark.parametrize("seed", range(8))
def test_infeasible_and_unbounded_solves_are_bit_identical(seed, kind, variant):
    problem = random_qp(
        seed, infeasible=(kind == "infeasible"), unbounded=(kind == "unbounded")
    )
    _, bad = _one_solve(problem, VARIANTS[variant])
    assert bad == []


def test_sweep_reaches_every_reported_status_family():
    seen = set()
    for seed in range(8):
        for kind in ("feasible", "infeasible", "unbounded"):
            problem = random_qp(
                seed, infeasible=(kind == "infeasible"), unbounded=(kind == "unbounded")
            )
            for variant in (0, 18):
                result, bad = _one_solve(problem, VARIANTS[variant])
                assert bad == []
                seen.add(result["status_val"])
    # solved, primal infeasible, dual infeasible, max iterations
    assert {1, 3, 5, 7} <= seen


@pytest.mark.parametrize("max_iter", [1, 2, 3, 5, 8, 13, 21, 34, 55])
@pytest.mark.parametrize("scaling", [0, 10])
def test_every_iteration_prefix_is_bit_identical(max_iter, scaling):
    """Stopping after k iterations exposes iterate k itself (scaled x, z, y)."""

    for seed in (1, 5):
        _, bad = _one_solve(
            random_qp(seed),
            {"max_iter": max_iter, "scaling": scaling, "check_termination": 0,
             "adaptive_rho_interval": 4},
        )
        assert bad == []


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"polishing": 1},
        {"scaling": 0, "adaptive_rho_interval": 9},
        {"warm_starting": 0},
        {"max_iter": 30, "check_termination": 5},
        {"adaptive_rho": 3, "adaptive_rho_interval": 0},
    ],
)
@pytest.mark.parametrize("seed", range(4))
def test_repeated_solves_with_vector_updates_are_bit_identical(seed, overrides):
    """osqp_update_data_vec + re-solve on one solver versus one Torch workspace."""

    rng = np.random.default_rng(900 + seed)
    P, q, A, l, u = random_qp(seed, m=int(rng.integers(2, 16)))
    n, m = len(q), len(l)
    c_settings, torch_settings = paired_settings(**overrides)
    workspace = TorchOSQPWorkspace()
    qk, lk, uk = q.copy(), l.copy(), u.copy()
    with PinnedOSQPSession(P, qk, A, lk, uk, **c_settings) as session:
        assert session.setup_code == 0
        for step, action in enumerate(
            ["initial", "q", "none", "bounds", "infeasible", "restore", "both"]
        ):
            new_q = new_l = new_u = None
            if action in ("q", "both"):
                qk = qk + 0.3 * rng.standard_normal(n)
                new_q = qk
            if action in ("bounds", "both"):
                shift = 0.05 * rng.standard_normal(m)
                lk = np.where(lk > -1e29, lk + shift, lk)
                uk = np.where(uk < 1e29, uk + shift, uk)
                new_l, new_u = lk, uk
            if action == "infeasible":
                lk, uk = lk.copy(), uk.copy()
                lk[0], uk[0] = 1e6, 1e6 + 1.0
                lk[1], uk[1] = -1e6 - 1.0, -1e6
                new_l, new_u = lk, uk
            if action == "restore":
                lk, uk = l.copy(), u.copy()
                new_l, new_u = lk, uk
            if new_q is not None or new_l is not None:
                assert session.update(q=new_q, l=new_l, u=new_u) == 0
            c_result = session.solve()
            c_work = read_workspace(session._solver)
            x, info = torch_solve((P, qk, A, lk, uk), torch_settings, workspace)
            # C keeps status_polish from an earlier solve; compare it on the first only.
            bad = compare_solve(
                c_result, c_work, x, info, workspace
            )
            if c_result["status_val"] not in (3, 4, 5, 6, 9):
                bad += compare_factors(c_work, workspace)
            assert bad == [], (step, action)


def test_explicit_warm_start_matches_osqp_warm_start():
    P, q, A, l, u = random_qp(3, m=9)
    c_settings, torch_settings = paired_settings(eps_abs=1e-7, eps_rel=1e-7)
    workspace = TorchOSQPWorkspace()
    with PinnedOSQPSession(P, q, A, l, u, **c_settings) as session:
        first = session.solve()
        x, info = torch_solve((P, q, A, l, u), torch_settings, workspace)
        assert same_bits(first["x"], x)
        x0 = first["x"] * 0.9 + 0.01
        y0 = first["y"] * 1.1
        assert session.warm_start(x=x0, y=y0) == 0
        second = session.solve()
        c_work = read_workspace(session._solver)
        workspace.state = {
            "x": torch.as_tensor(x0), "y": torch.as_tensor(y0),
            "z": torch.zeros(len(l), dtype=torch.float64),
        }
        x, info = torch_solve((P, q, A, l, u), torch_settings, workspace)
    # osqp_warm_start does not reset info, so the rho update counter accumulates in
    # C; the Torch workspace keeps it too because the problem data is unchanged.
    assert compare_solve(second, c_work, x, info, workspace) == []


def test_qdldl_factorization_matches_c_factors_on_c_kkt():
    """QDLDL_etree / QDLDL_factor on C's own permuted KKT reproduce C's L, D, Dinv."""

    compared = 0
    for seed in range(40):
        problem = random_qp(seed)
        n, m = len(problem[1]), len(problem[3])
        c_settings, _ = paired_settings(scaling=int(seed % 2) * 10)
        with PinnedOSQPSession(*problem, **c_settings) as session:
            assert session.setup_code == 0
            lin = read_workspace(session._solver)["linsys"]
        kkt = lin["KKT"]
        size = n + m
        total, Lnz, etree = qdldl_etree(size, kkt["p"].tolist(), kkt["i"].tolist())
        assert Lnz == lin["Lnz"].tolist() and etree == lin["etree"].tolist()
        positive, Lp, Li, Lx, D, Dinv = qdldl_factor(
            size, kkt["p"].tolist(), kkt["i"].tolist(), kkt["x"].tolist(),
            Lnz, etree, total,
        )
        assert positive == n
        assert Lp == lin["L"]["p"].tolist() and Li == lin["L"]["i"].tolist()
        assert np.asarray(Lx).tobytes() == lin["L"]["x"].tobytes()
        assert np.asarray(D).tobytes() == lin["D"].tobytes()
        assert np.asarray(Dinv).tobytes() == lin["Dinv"].tobytes()
        compared += 1
    assert compared >= 35


def test_sequential_sum_is_the_c_accumulation_recurrence():
    """``_seq_sum`` must equal ``acc += term`` in index order, bit for bit."""

    generator = torch.Generator().manual_seed(7)
    for size in (0, 1, 2, 7, 8, 9, 31, 32, 33, 257, 1000):
        values = torch.randn(size, dtype=torch.float64, generator=generator)
        values = values * torch.exp(4 * torch.randn(size, dtype=torch.float64, generator=generator))
        expected = 0.0
        for term in values.tolist():
            expected += term
        assert np.float64(torchOSQP._seq_sum(values).item()).tobytes() == np.float64(expected).tobytes()
    matrix = torch.randn(13, 29, dtype=torch.float64, generator=generator)
    vector = torch.randn(29, dtype=torch.float64, generator=generator)
    expected = []
    for row in matrix.tolist():
        total = 0.0
        for entry, x in zip(row, vector.tolist(), strict=True):
            total += entry * x
        expected.append(total)
    assert torchOSQP._matvec(matrix, vector).tolist() == expected


def test_c_sqrt_is_correctly_rounded_where_torch_sqrt_is_not_guaranteed():
    generator = torch.Generator().manual_seed(11)
    values = torch.rand(20000, dtype=torch.float64, generator=generator) * 100
    assert torchOSQP._c_sqrt(values).tolist() == [math.sqrt(v) for v in values.tolist()]
    scalar = torch.tensor(0.5745171192538397, dtype=torch.float64)
    assert torchOSQP._c_sqrt(scalar).item() == math.sqrt(scalar.item())


@pytest.mark.parametrize("kind", ["infeasible", "unbounded"])
@pytest.mark.parametrize("seed", range(6))
def test_status_carried_over_from_an_earlier_solve_is_reported_as_c_does(seed, kind):
    """Re-solving unchanged data keeps C's stored status, objective and certificate.

    ``osqp_solve`` does not reset ``info->status_val``.  When a shorter re-solve
    does not detect infeasibility again, C keeps the old status, reports the
    objective from the last ``update_info`` and normalizes the current
    ``delta_y`` / ``delta_x`` in ``store_solution``.
    """

    problem = random_qp(
        seed, m=8, infeasible=(kind == "infeasible"), unbounded=(kind == "unbounded")
    )
    c_settings, torch_settings = paired_settings()
    workspace = TorchOSQPWorkspace()
    with PinnedOSQPSession(*problem, **c_settings) as session:
        assert session.setup_code == 0
        first = session.solve()
        c_work = read_workspace(session._solver)
        x, info = torch_solve(problem, torch_settings, workspace)
        assert compare_solve(first, c_work, x, info, workspace) == []
        if first["status_val"] not in (3, 4, 5, 6):
            pytest.skip("the first solve did not end infeasible")
        for max_iter in (1, 2, 7, 4000):
            # Same effect as osqp_update_settings(max_iter): no info reset.
            session._solver.contents.settings.contents.max_iter = max_iter
            c_result = session.solve()
            c_work = read_workspace(session._solver)
            x, info = torch_solve(
                problem, dict(torch_settings, max_iter=max_iter), workspace
            )
            assert compare_solve(
                c_result, c_work, x, info, workspace
            ) == [], max_iter


@pytest.mark.parametrize(
    "overrides",
    [{}, {"scaling": 0}, {"polishing": 1}, {"scaling": 2, "adaptive_rho_interval": 9},
     {"warm_starting": 0}],
)
@pytest.mark.parametrize("seed", range(5))
def test_matrix_value_updates_are_bit_identical(seed, overrides):
    """``osqp_update_data_mat`` (unscale, insert values, rescale) then re-solve.

    Convention for one Torch call that changes several inputs: the matrix
    update is applied first, then the vector update, as in the C calls below.
    """

    rng = np.random.default_rng(300 + seed)
    P, q, A, l, u = random_qp(seed, m=int(rng.integers(1, 14)))
    n, m = len(q), len(l)
    c_settings, torch_settings = paired_settings(**overrides)
    workspace = TorchOSQPWorkspace()
    Pk, qk, Ak, lk, uk = P.copy(), q.copy(), A.copy(), l.copy(), u.copy()
    with PinnedOSQPSession(P, q, A, l, u, **c_settings) as session:
        assert session.setup_code == 0
        for step, action in enumerate(["initial", "P", "A", "none", "PA", "P+q", "A+lu", "q"]):
            head = action.split("+")[0]
            new_P = new_A = None
            if head in ("P", "PA"):
                Pk = Pk * (1.0 + 0.2 * rng.random())
                new_P = Pk
            if head in ("A", "PA"):
                Ak = Ak * (1.0 + 0.1 * rng.standard_normal((m, 1)))
                new_A = Ak
            if new_P is not None or new_A is not None:
                assert session.update_matrices(P=new_P, A=new_A) == 0
            if action in ("P+q", "q"):
                qk = qk + 0.2 * rng.standard_normal(n)
                assert session.update(q=qk) == 0
            if action == "A+lu":
                shift = 0.05 * rng.standard_normal(m)
                lk = np.where(lk > -1e29, lk + shift, lk)
                uk = np.where(uk < 1e29, uk + shift, uk)
                assert session.update(l=lk, u=uk) == 0
            c_result = session.solve()
            c_work = read_workspace(session._solver)
            x, info = torch_solve((Pk, qk, Ak, lk, uk), torch_settings, workspace)
            bad = compare_solve(
                c_result, c_work, x, info, workspace
            )
            if c_result["status_val"] not in (3, 4, 5, 6, 9):
                bad += compare_factors(c_work, workspace)
            assert bad == [], (step, action)
