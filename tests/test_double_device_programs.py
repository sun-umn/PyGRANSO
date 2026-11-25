"""The tensor programs of ``torchQDLDLDevice`` against the scalar QDLDL port.

CUDA tensors are factorized and solved by these programs so that no value
leaves the GPU.  Every result must have the bits of the scalar loops.
"""

import numpy as np
import pytest
import torch

import pygranso.private.torchCSC as torch_csc
import pygranso.private.torchQDLDLDevice as device_programs
from pygranso.private.torchQDLDL import ldl_solve, qdldl_etree, qdldl_factor
from pygranso.private.torchQDLDLDevice import FactorProgram, SolveProgram

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _bits(values):
    if torch.is_tensor(values):
        values = values.detach().cpu()
    return np.asarray(values, dtype="<f8").tobytes()


def _matrix(seed):
    """Upper-triangular CSC lists of a random symmetric matrix with full diagonal."""

    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    density = rng.choice([0.05, 0.2, 0.6, 1.0])
    M = np.triu(rng.standard_normal((n, n)) * (rng.random((n, n)) < density), 1)
    diagonal = rng.standard_normal(n) * 3
    diagonal[diagonal == 0] = 1.0
    if seed % 5 == 0 and n > 2:
        diagonal[int(rng.integers(0, n))] = 0.0  # can give a zero entry of D
    M = M + np.diag(diagonal)
    stored = (M != 0) | np.eye(n, dtype=bool) | np.triu(rng.random((n, n)) < 0.05)
    Ap, Ai, Ax = [0], [], []
    for j in range(n):
        for i in range(j + 1):
            if stored[i, j]:
                Ai.append(i)
                Ax.append(float(M[i, j]))
        Ap.append(len(Ai))
    return n, Ap, Ai, Ax, list(rng.permutation(n)), rng.standard_normal(n)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("forward_by_rows", [None, True, False])
@pytest.mark.parametrize("seed", range(40))
def test_programs_have_the_bits_of_the_scalar_loops(
    seed, forward_by_rows, loop, device, monkeypatch
):
    monkeypatch.setattr(torch_csc, "FORCE_LEVEL_LOOP", loop)
    monkeypatch.setattr(device_programs, "FORWARD_BY_ROWS", forward_by_rows)
    n, Ap, Ai, Ax, perm, b = _matrix(seed)
    sumLnz, Lnz, etree = qdldl_etree(n, Ap, Ai)
    positive, Lp, Li, Lx, D, Dinv = qdldl_factor(n, Ap, Ai, Ax, Lnz, etree, sumLnz)

    program = FactorProgram(n, Ap, Ai, Lnz, etree, sumLnz, device)
    t_Lx, t_D, t_Dinv = program.run(torch.tensor(Ax, dtype=torch.float64, device=device))
    assert t_Lx.device.type == device
    assert bool((t_D == 0).any()) == (positive < 0)
    if positive < 0:
        return
    assert (program.Lp, program.Li) == (Lp, Li)
    assert int((t_D > 0).sum()) == positive
    assert _bits(t_Lx) == _bits(Lx)
    assert _bits(t_D) == _bits(D)
    assert _bits(t_Dinv) == _bits(Dinv)

    solve = SolveProgram(n, Lp, Li, perm, device)
    x = solve.run(
        solve.bind(t_Lx), t_Dinv, torch.tensor(b, dtype=torch.float64, device=device)
    )
    assert x.device.type == device
    assert _bits(x) == _bits(ldl_solve(list(b), Lp, Li, Lx, Dinv, perm))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("rows", [1, 2, 3, 31, 257, 4097, 100000])
@pytest.mark.parametrize("columns", [2, 3, 40])
def test_sequential_sum_adds_the_terms_in_order(columns, rows, in_place, device):
    """Every sum of the device programs relies on this order of ``cumsum``."""

    generator = torch.Generator().manual_seed(rows * 1000 + columns)
    # Magnitudes differ widely, so another order of addition changes the bits.
    values = (torch.rand(rows, columns, generator=generator, dtype=torch.float64) - 0.5) * (
        torch.exp(6 * torch.randn(rows, columns, generator=generator, dtype=torch.float64))
    )
    values[0, 0] = -0.0
    values[:, 1] = -0.0  # C keeps -0.0
    if columns > 2:
        values[:, 2] = -0.0  # one +0.0 term makes the sum +0.0
        values[rows // 2, 2] = 0.0
    expected = [float(v) for v in values[0]]
    for row in values[1:].tolist():
        expected = [total + term for total, term in zip(expected, row)]
    buffer = values.to(device)
    result = (
        torch_csc.sequential_sum_(buffer) if in_place else torch_csc.sequential_sum(buffer)
    )
    assert result.device.type == device
    assert _bits(result) == _bits(expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("forward_by_rows", [None, True, False])
@pytest.mark.parametrize("seed", range(40))
def test_solve_keeps_the_sign_of_zero(seed, forward_by_rows, loop, device, monkeypatch):
    """Right-hand sides of signed zeros: C can return -0.0 and so must the program."""

    monkeypatch.setattr(torch_csc, "FORCE_LEVEL_LOOP", loop)
    monkeypatch.setattr(device_programs, "FORWARD_BY_ROWS", forward_by_rows)
    n, Ap, Ai, Ax, perm, b = _matrix(seed)
    rng = np.random.default_rng(1000 + seed)
    kind = rng.integers(0, 3 if seed % 2 else 2, n)  # -0.0, +0.0, sometimes a value
    b = np.where(kind == 0, -0.0, np.where(kind == 1, 0.0, b))
    Ax = [abs(v) if seed % 4 < 2 else v for v in Ax]  # also all-positive factors
    sumLnz, Lnz, etree = qdldl_etree(n, Ap, Ai)
    positive, Lp, Li, Lx, D, Dinv = qdldl_factor(n, Ap, Ai, Ax, Lnz, etree, sumLnz)
    if positive < 0:
        return
    solve = SolveProgram(n, Lp, Li, perm, device)
    x = solve.run(
        solve.bind(torch.tensor(Lx, dtype=torch.float64, device=device)),
        torch.tensor(Dinv, dtype=torch.float64, device=device),
        torch.tensor(b, dtype=torch.float64, device=device),
    )
    assert _bits(x) == _bits(ldl_solve(list(b), Lp, Li, Lx, Dinv, perm))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_accumulate_from_a_start_keeps_the_sign_of_zero(seed, loop, device, monkeypatch):
    """``y[i] += M[i, j] * x[j]`` from an existing ``y`` with signed zeros."""

    monkeypatch.setattr(torch_csc, "FORCE_LEVEL_LOOP", loop)
    rng = np.random.default_rng(seed)
    n_out, n_in = int(rng.integers(1, 12)), int(rng.integers(1, 12))
    stored = rng.random((n_out, n_in)) < 0.4
    out, inp = np.nonzero(stored)
    signed = np.array([-0.0, 0.0, 1.5, -2.0])
    val = rng.choice(signed, out.size)
    x = rng.choice(signed, n_in)
    start = rng.choice(signed[:3], n_out)
    plan = torch_csc._Plan(
        torch.tensor(out, device=device),
        torch.tensor(inp, device=device),
        torch.tensor(val, dtype=torch.float64, device=device),
        n_out,
        n_in,
    )
    t_x = torch.tensor(x, dtype=torch.float64, device=device)
    for subtract in (False, True):
        for use_start in (True, False):
            expected = [float(v) for v in start] if use_start else [0.0] * n_out
            for i, j, v in sorted(zip(out.tolist(), inp.tolist(), val.tolist())):
                term = v * float(x[j])
                expected[i] = expected[i] - term if subtract else expected[i] + term
            products = plan.val * t_x[plan.inp]
            result = plan.accumulate(
                -products if subtract else products,
                torch.tensor(start, dtype=torch.float64, device=device) if use_start else None,
            )
            assert _bits(result) == _bits(expected)
