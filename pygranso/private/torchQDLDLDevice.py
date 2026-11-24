"""``QDLDL_factor`` and ``QDLDL_solve`` as tensor operations on the data's device.

``torchQDLDL.py`` runs the QDLDL loops one scalar at a time.  This module
performs the same scalar operations, each with the same operands and in the same
order per result, but grouped so that operations that do not depend on each
other are done together by one tensor operation.  The values never leave the
device of the input.

The grouping uses only the sparsity pattern (integers), so it is computed once
per pattern by walking the loops of ``QDLDL_factor``, ``QDLDL_Lsolve`` and
``QDLDL_Ltsolve`` without values:

* Every number QDLDL computes is ``start - t1 - t2 - ...`` (``yVals``, ``D``,
  the solve vectors), a product (``Lx``) or a reciprocal (``Dinv``).  The terms
  of one result are subtracted in the C order.
* A term is applied in the first step in which its two factors and the previous
  term of the same result are available.  All terms of one step are independent.
* In ``QDLDL_Ltsolve`` all factors of a row become available together, so the
  whole row is summed at once by ``torchCSC.sequential_sum``.

A multiplication and the following subtraction are separate operations, as in C
built without floating-point contraction.
"""

from __future__ import annotations

import torch

import pygranso.private.torchCSC as torch_csc
from pygranso.private.torchQDLDL import QDLDL_UNKNOWN, QDLDL_UNUSED, QDLDL_USED


def _steps(times, columns, device):
    """Group operations by step: a list of tuples of index tensors, step 1 first."""

    if not times:
        return []
    time = torch.tensor(times, dtype=torch.int64)
    order = torch.argsort(time, stable=True)
    sizes = torch.bincount(time)[1:].tolist()
    parts = [
        torch.split(torch.tensor(column, dtype=torch.int64)[order].to(device), sizes)
        for column in columns
    ]
    return list(zip(*parts))


class FactorProgram:
    """The operations of ``QDLDL_factor`` for one upper-triangular CSC pattern."""

    def __init__(self, n, Ap, Ai, Lnz, etree, sumLnz, device):
        # Value slots: the entries of A, a zero, then yVals (one per entry of
        # L), Lx, D and Dinv.
        nnz = Ap[n]
        zero_slot = nnz
        y0 = nnz + 1
        l0 = y0 + sumLnz
        d0 = l0 + sumLnz
        i0 = d0 + n
        self.size = i0 + n
        self.n, self.nnz, self.sumLnz = n, nnz, sumLnz
        self.l0, self.d0, self.i0 = l0, d0, i0

        ready = [0] * self.size
        start_target, start_source = [], []
        sub = ([], [], [], [])  # step, target, factor, factor
        mul = ([], [], [], [])
        rec = ([], [], [])  # step, target, source

        Lp = [0] * (n + 1)
        Li = [0] * sumLnz
        yMarkers = [QDLDL_UNUSED] * n
        yIdx = [0] * n
        elimBuffer = [0] * n
        LNextSpaceInCol = [0] * n
        ySource = [zero_slot] * n
        for i in range(n):
            Lp[i + 1] = Lp[i] + Lnz[i]
            LNextSpaceInCol[i] = Lp[i]

        def reciprocal(k):
            step = ready[d0 + k] + 1
            ready[i0 + k] = step
            rec[0].append(step)
            rec[1].append(i0 + k)
            rec[2].append(d0 + k)

        # D[0] = Ax[0]
        start_target.append(d0)
        start_source.append(0)
        reciprocal(0)

        for k in range(1, n):
            nnzY = 0
            d_source = zero_slot
            for i in range(Ap[k], Ap[k + 1]):
                bidx = Ai[i]
                if bidx == k:
                    d_source = i
                    continue
                ySource[bidx] = i
                nextIdx = bidx
                if yMarkers[nextIdx] == QDLDL_UNUSED:
                    yMarkers[nextIdx] = QDLDL_USED
                    elimBuffer[0] = nextIdx
                    nnzE = 1
                    nextIdx = etree[bidx]
                    while nextIdx != QDLDL_UNKNOWN and nextIdx < k:
                        if yMarkers[nextIdx] == QDLDL_USED:
                            break
                        yMarkers[nextIdx] = QDLDL_USED
                        elimBuffer[nnzE] = nextIdx
                        nnzE += 1
                        nextIdx = etree[nextIdx]
                    while nnzE:
                        nnzE -= 1
                        yIdx[nnzY] = elimBuffer[nnzE]
                        nnzY += 1

            d_slot = d0 + k
            start_target.append(d_slot)
            start_source.append(d_source)
            for i in range(nnzY - 1, -1, -1):
                cidx = yIdx[i]
                tmpIdx = LNextSpaceInCol[cidx]
                y_slot = y0 + tmpIdx
                start_target.append(y_slot)
                start_source.append(ySource[cidx])
                y_ready = ready[y_slot]
                for j in range(Lp[cidx], tmpIdx):
                    # yVals[Li[j]] -= Lx[j] * yVals_cidx
                    row = Li[j]
                    if yMarkers[row] != QDLDL_USED:
                        raise RuntimeError("QDLDL elimination order is inconsistent.")
                    target = y0 + LNextSpaceInCol[row]
                    step = max(ready[l0 + j], y_ready, ready[target]) + 1
                    ready[target] = step
                    sub[0].append(step)
                    sub[1].append(target)
                    sub[2].append(l0 + j)
                    sub[3].append(y_slot)
                Li[tmpIdx] = k
                # Lx[tmpIdx] = yVals_cidx * Dinv[cidx]
                l_slot = l0 + tmpIdx
                step = max(y_ready, ready[i0 + cidx]) + 1
                ready[l_slot] = step
                mul[0].append(step)
                mul[1].append(l_slot)
                mul[2].append(y_slot)
                mul[3].append(i0 + cidx)
                # D[k] -= yVals_cidx * Lx[tmpIdx]
                step = max(step, ready[d_slot]) + 1
                ready[d_slot] = step
                sub[0].append(step)
                sub[1].append(d_slot)
                sub[2].append(y_slot)
                sub[3].append(l_slot)
                LNextSpaceInCol[cidx] += 1
                ySource[cidx] = zero_slot
                yMarkers[cidx] = QDLDL_UNUSED
            reciprocal(k)

        self.Lp, self.Li = Lp, Li
        self.start_target = torch.tensor(start_target, dtype=torch.int64, device=device)
        self.start_source = torch.tensor(start_source, dtype=torch.int64, device=device)
        last = max(ready)
        sub_steps = _steps(sub[0], sub[1:], device)
        mul_steps = _steps(mul[0], mul[1:], device)
        rec_steps = _steps(rec[0], rec[1:], device)
        self.steps = [
            tuple(
                group[index] if index < len(group) and group[index][0].numel() else None
                for group in (sub_steps, mul_steps, rec_steps)
            )
            for index in range(last)
        ]

    def run(self, Ax):
        """Returns ``(Lx, D, Dinv)`` for the values ``Ax`` of the pattern."""

        values = Ax.new_zeros(self.size)
        values[: self.nnz] = Ax
        values[self.start_target] = values[self.start_source]
        one = Ax.new_ones(())
        for sub, mul, rec in self.steps:
            if sub is not None:
                target, a, b = sub
                values[target] = values[target] - values[a] * values[b]
            if mul is not None:
                target, a, b = mul
                values[target] = values[a] * values[b]
            if rec is not None:
                target, source = rec
                values[target] = one / values[source]
        return (
            values[self.l0 : self.l0 + self.sumLnz].clone(),
            values[self.d0 : self.d0 + self.n].clone(),
            values[self.i0 : self.i0 + self.n].clone(),
        )


class SolveProgram:
    """The operations of ``LDLSolve`` for the pattern of ``L`` and a permutation."""

    def __init__(self, n, Lp, Li, perm, device):
        self.n = n
        self.perm = torch.tensor(perm, dtype=torch.int64, device=device)

        # QDLDL_Lsolve: x[Li[j]] -= Lx[j] * x[i], i ascending.
        ready = [0] * n
        forward = ([], [], [], [])  # step, target, entry of L, source
        for i in range(n):
            ready_i = ready[i]
            for j in range(Lp[i], Lp[i + 1]):
                row = Li[j]
                step = max(ready_i, ready[row]) + 1
                ready[row] = step
                forward[0].append(step)
                forward[1].append(row)
                forward[2].append(j)
                forward[3].append(i)
        self.forward = _steps(forward[0], forward[1:], device)

        # QDLDL_Ltsolve: x[i] -= Lx[j] * x[Li[j]] over column i, i descending.
        # A row is computed once every x[Li[j]] it uses is final.  The rows of
        # one stage are summed together; when their lengths differ so much that
        # the padded buffer would be mostly padding, they are split by length.
        stage = [0] * n
        stages = {}
        for i in range(n - 1, -1, -1):
            count = Lp[i + 1] - Lp[i]
            if count == 0:
                continue
            stage[i] = 1 + max(stage[Li[j]] for j in range(Lp[i], Lp[i + 1]))
            stages.setdefault(stage[i], []).append(i)
        groups = {}
        for key in sorted(stages):
            rows = stages[key]
            counts = [Lp[i + 1] - Lp[i] for i in rows]
            if (max(counts) + 1) * (len(rows) + 1) <= 4 * sum(counts) + 1024:
                groups[(key, 0)] = rows
                continue
            for i, count in zip(rows, counts):
                groups.setdefault((key, (count - 1).bit_length() + 1), []).append(i)
        self.backward = []
        for key in sorted(groups):
            rows = groups[key]
            width = max(Lp[i + 1] - Lp[i] for i in rows)
            entry, source = [], []
            for i in rows:
                entry.extend(range(Lp[i], Lp[i + 1]))
                source.extend(Li[Lp[i] : Lp[i + 1]])
            # The terms of the group are read from [x, products, -0.0]: row 0 of
            # the buffer is x[rows], row r the r-th product of each row, and
            # -0.0 elsewhere (its addition changes nothing).  The spare last
            # column is required by sequential_sum.
            padding = n + len(entry)
            pick = [[padding] * (len(rows) + 1) for _ in range(width + 1)]
            offset = n
            for column, i in enumerate(rows):
                pick[0][column] = i
                for rank in range(Lp[i + 1] - Lp[i]):
                    pick[rank + 1][column] = offset + rank
                offset += Lp[i + 1] - Lp[i]
            self.backward.append(
                (
                    torch.tensor(rows, dtype=torch.int64, device=device),
                    torch.tensor(pick, dtype=torch.int64, device=device),
                    torch.tensor(entry, dtype=torch.int64, device=device),
                    torch.tensor(source, dtype=torch.int64, device=device),
                )
            )

    def bind(self, Lx):
        """Gather ``-Lx`` in the order the steps use it.

        ``a - l * b`` is computed as ``a + (-l) * b``: negation is exact, so the
        product and the sum round as in C.
        """

        negative = -Lx
        return (
            [negative[entry] for _, entry, _ in self.forward],
            [negative[entry] for _, _, entry, _ in self.backward],
            Lx.new_full((1,), -0.0),
        )

    def run(self, bound, Dinv, b):
        forward_values, backward_values, padding = bound
        # bp[j] = b[P[j]]
        x = b[self.perm]
        for (target, _, source), negative in zip(self.forward, forward_values):
            # x[Li[j]] -= Lx[j] * x[i]; each target occurs once per step.
            x.index_add_(0, target, negative * x[source])
        # x[i] *= Dinv[i]
        x = x * Dinv
        for (rows, pick, _, source), negative in zip(self.backward, backward_values):
            # val -= Lx[j] * x[Li[j]], j ascending, for every row of the group.
            buffer = torch.cat((x, negative * x[source], padding))[pick]
            if torch_csc.FORCE_LEVEL_LOOP or x.dtype != torch.float64:
                total = buffer[0].clone()
                for rank in range(1, buffer.shape[0]):
                    total = total + buffer[rank]
            else:
                total = torch_csc.sequential_sum(buffer)
            x[rows] = total[:-1]
        # x[P[j]] = bp[j]
        result = torch.empty_like(x)
        result[self.perm] = x
        return result
