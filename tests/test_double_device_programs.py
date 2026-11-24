"""The tensor programs of ``torchQDLDLDevice`` against the scalar QDLDL port.

CUDA tensors are factorized and solved by these programs so that no value
leaves the GPU.  Every result must have the bits of the scalar loops.
"""

import numpy as np
import pytest
import torch

import pygranso.private.torchCSC as torch_csc
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
@pytest.mark.parametrize("seed", range(40))
def test_programs_have_the_bits_of_the_scalar_loops(seed, loop, device, monkeypatch):
    monkeypatch.setattr(torch_csc, "FORCE_LEVEL_LOOP", loop)
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
