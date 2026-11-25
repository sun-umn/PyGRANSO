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
* ``QDLDL_Lsolve`` subtracts the terms of ``x[r]`` in ascending column order,
  each with the final ``x[i]`` of its column.  It is either applied term by
  term as above, or summed row by row like ``QDLDL_Ltsolve``; the choice only
  changes how many tensor operations are needed.

A multiplication and the following subtraction are separate operations, as in C
built without floating-point contraction.
"""

from __future__ import annotations

import numpy as np
import torch

import pygranso.private.torchCSC as torch_csc
from pygranso.private.torchQDLDL import QDLDL_UNKNOWN, QDLDL_UNUSED, QDLDL_USED

# Test hook for ``QDLDL_Lsolve``: ``True`` sums it row by row, ``False`` applies
# it step by step, ``None`` takes the one with fewer tensor operations.  The
# three give the same bits.
FORWARD_BY_ROWS = None

# Columns of L up to this length are scheduled one term at a time in the setup
# of FactorProgram; longer ones with array operations.  Only the setup time
# depends on it.
_SHORT_COLUMN = 16


def _steps(times, columns, device):
    """Group operations by step: a list of tuples of index tensors, step 1 first."""

    if len(times) == 0:
        return []
    time = torch.as_tensor(times, dtype=torch.int64)
    order = torch.argsort(time, stable=True)
    sizes = torch.bincount(time)[1:].tolist()
    parts = [
        torch.split(
            torch.as_tensor(column, dtype=torch.int64)[order].to(device), sizes
        )
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

        # ready[slot] is the step after which a value is final.  The loop over
        # one column of L (the innermost loop of QDLDL_factor) touches every
        # target once, so its steps are computed for the whole column at once.
        ready = np.zeros(self.size, dtype=np.int64)
        start_target, start_source = [], []
        column_steps, column_targets = [], []  # yVals[Li[j]] -= Lx[j] * yVals_cidx
        column_first, column_count, column_y = [], [], []
        d_sub = ([], [], [], [])  # step, target, factor, factor
        mul = ([], [], [], [])
        rec = ([], [], [])  # step, target, source

        Lp = [0] * (n + 1)
        Li = np.zeros(sumLnz, dtype=np.int64)
        Li_list = [0] * sumLnz  # the same, for the scalar loop of short columns
        yMarkers = [QDLDL_UNUSED] * n
        used = np.zeros(n, dtype=bool)
        yIdx = [0] * n
        elimBuffer = [0] * n
        LNextSpaceInCol = np.zeros(n, dtype=np.int64)
        ySource = [zero_slot] * n
        for i in range(n):
            Lp[i + 1] = Lp[i] + Lnz[i]
            LNextSpaceInCol[i] = Lp[i]
        next_list = Lp[:n]  # LNextSpaceInCol, for scalar access

        def reciprocal(k):
            step = int(ready[d0 + k]) + 1
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
                    used[nextIdx] = True
                    elimBuffer[0] = nextIdx
                    nnzE = 1
                    nextIdx = etree[bidx]
                    while nextIdx != QDLDL_UNKNOWN and nextIdx < k:
                        if yMarkers[nextIdx] == QDLDL_USED:
                            break
                        yMarkers[nextIdx] = QDLDL_USED
                        used[nextIdx] = True
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
            d_ready = 0
            for i in range(nnzY - 1, -1, -1):
                cidx = yIdx[i]
                tmpIdx = next_list[cidx]
                y_slot = y0 + tmpIdx
                start_target.append(y_slot)
                start_source.append(ySource[cidx])
                y_ready = int(ready[y_slot])
                first = Lp[cidx]
                if tmpIdx - first <= _SHORT_COLUMN:
                    # yVals[Li[j]] -= Lx[j] * yVals_cidx, one term at a time
                    for j in range(first, tmpIdx):
                        row = Li_list[j]
                        if yMarkers[row] != QDLDL_USED:
                            raise RuntimeError("QDLDL elimination order is inconsistent.")
                        target = y0 + next_list[row]
                        step = max(ready.item(l0 + j), y_ready, ready.item(target)) + 1
                        ready[target] = step
                        d_sub[0].append(step)
                        d_sub[1].append(target)
                        d_sub[2].append(l0 + j)
                        d_sub[3].append(y_slot)
                else:
                    # The same for j = first .. tmpIdx - 1 at once.
                    rows = Li[first:tmpIdx]
                    if not used[rows].all():
                        raise RuntimeError("QDLDL elimination order is inconsistent.")
                    targets = y0 + LNextSpaceInCol[rows]
                    steps = (
                        np.maximum(
                            np.maximum(ready[l0 + first : l0 + tmpIdx], y_ready),
                            ready[targets],
                        )
                        + 1
                    )
                    ready[targets] = steps
                    column_steps.append(steps)
                    column_targets.append(targets)
                    column_first.append(first)
                    column_count.append(tmpIdx - first)
                    column_y.append(y_slot)
                Li[tmpIdx] = k
                Li_list[tmpIdx] = k
                # Lx[tmpIdx] = yVals_cidx * Dinv[cidx]
                l_slot = l0 + tmpIdx
                step = max(y_ready, int(ready[i0 + cidx])) + 1
                ready[l_slot] = step
                mul[0].append(step)
                mul[1].append(l_slot)
                mul[2].append(y_slot)
                mul[3].append(i0 + cidx)
                # D[k] -= yVals_cidx * Lx[tmpIdx]
                d_ready = max(step, d_ready) + 1
                d_sub[0].append(d_ready)
                d_sub[1].append(d_slot)
                d_sub[2].append(y_slot)
                d_sub[3].append(l_slot)
                LNextSpaceInCol[cidx] += 1
                next_list[cidx] += 1
                ySource[cidx] = zero_slot
                yMarkers[cidx] = QDLDL_UNUSED
                used[cidx] = False
            ready[d_slot] = d_ready
            reciprocal(k)

        self.Lp, self.Li = Lp, Li_list
        self.start_target = torch.tensor(start_target, dtype=torch.int64, device=device)
        self.start_source = torch.tensor(start_source, dtype=torch.int64, device=device)
        # All subtractions: those of the long columns, then the others.
        counts = np.asarray(column_count, dtype=np.int64)
        offsets = np.cumsum(counts) - counts
        within = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(offsets, counts)
        sub = (
            np.concatenate(column_steps + [np.asarray(d_sub[0], dtype=np.int64)]),
            np.concatenate(column_targets + [np.asarray(d_sub[1], dtype=np.int64)]),
            np.concatenate(
                (
                    l0 + np.repeat(np.asarray(column_first, dtype=np.int64), counts) + within,
                    np.asarray(d_sub[2], dtype=np.int64),
                )
            ),
            np.concatenate(
                (
                    np.repeat(np.asarray(column_y, dtype=np.int64), counts),
                    np.asarray(d_sub[3], dtype=np.int64),
                )
            ),
        )
        last = int(ready.max())
        # Per step: the factors of all products as one (2, k) index, the
        # products that are subtracted first, then the entries of L; and the
        # reciprocals.  Everything is moved to the device as six tensors.
        cpu = torch.device("cpu")
        sub_steps = _steps(sub[0], sub[1:], cpu)
        mul_steps = _steps(mul[0], mul[1:], cpu)
        rec_steps = _steps(rec[0], rec[1:], cpu)
        empty = torch.empty(0, dtype=torch.int64)

        def part(group, index, column):
            return group[index][column] if index < len(group) else empty

        flat = [[], [], [], [], []]  # factors, sub target, mul target, rec target, rec source
        sizes = [[], [], [], [], []]
        for index in range(last):
            pieces = (
                torch.cat(
                    (
                        part(sub_steps, index, 1),
                        part(mul_steps, index, 1),
                        part(sub_steps, index, 2),
                        part(mul_steps, index, 2),
                    )
                ),
                part(sub_steps, index, 0),
                part(mul_steps, index, 0),
                part(rec_steps, index, 0),
                part(rec_steps, index, 1),
            )
            for column, piece in enumerate(pieces):
                flat[column].append(piece)
                sizes[column].append(piece.numel())
        if last:
            parts = [
                torch.split(torch.cat(column).to(device), size)
                for column, size in zip(flat, sizes)
            ]
            self.steps = [
                (
                    factors.view(2, -1) if factors.numel() else None,
                    sub_target if sub_target.numel() else None,
                    mul_target if mul_target.numel() else None,
                    (rec_target, rec_source) if rec_target.numel() else None,
                )
                for factors, sub_target, mul_target, rec_target, rec_source in zip(*parts)
            ]
        else:
            self.steps = []

    def run(self, Ax):
        """Returns ``(Lx, D, Dinv)`` for the values ``Ax`` of the pattern."""

        values = Ax.new_zeros(self.size)
        values[: self.nnz] = Ax
        values[self.start_target] = values[self.start_source]
        one = Ax.new_ones(())
        for factors, sub_target, mul_target, rec in self.steps:
            if factors is not None:
                pair = values[factors]
                product = pair[0] * pair[1]
                count = 0
                if sub_target is not None:
                    # target -= product, as target += -1.0 * product: the
                    # multiplication by -1.0 is exact and each target occurs
                    # once per step.
                    count = sub_target.numel()
                    values.index_add_(0, sub_target, product[:count], alpha=-1.0)
                if mul_target is not None:
                    values[mul_target] = product[count:]
            if rec is not None:
                target, source = rec
                values[target] = one / values[source]
        return (
            values[self.l0 : self.l0 + self.sumLnz].clone(),
            values[self.d0 : self.d0 + self.n].clone(),
            values[self.i0 : self.i0 + self.n].clone(),
        )


class _RowSums:
    """Rows ``x[r] = x[r] + c1 * x[s1] + c2 * x[s2] + ...`` summed in that order.

    A row is computed in the first stage in which every ``x[s]`` it uses is
    final, and the rows of one stage are summed together.  ``x`` is followed by
    an entry equal to 1.0 and the coefficients by 1.0 and -0.0, so a stage is
    one product of two gathered tables: row 0 is ``x[r] * 1.0``, row ``k`` the
    ``k``-th term of each row, and ``1.0 * -0.0`` elsewhere (adding -0.0 changes
    nothing).  The spare last column is required by ``sequential_sum``; its sum
    is written to a second extra entry of ``x`` that nothing reads.
    """

    def __init__(self, n, count, stage, pointers, entries, sources, device):
        # stage[r] is 0 for a row without terms.  The terms of row r are
        # pointers[r] .. pointers[r + 1] - 1 of ``entries`` (the coefficients)
        # and ``sources`` (the entries of x), in summation order.
        lengths = np.diff(pointers)
        rows = np.flatnonzero(stage)
        rows = rows[np.argsort(stage[rows], kind="stable")]
        bounds = np.flatnonzero(np.diff(stage[rows])) + 1
        # When the lengths of the rows of a stage differ so much that the table
        # would be mostly padding, the rows are split by length.
        self.groups = []
        for group in np.split(rows, bounds) if rows.size else []:
            counts = lengths[group]
            if (int(counts.max()) + 1) * (group.size + 1) <= 4 * int(counts.sum()) + 1024:
                self.groups.append(group)
                continue
            classes = np.frexp(counts - 1)[1]  # bit length of the length - 1
            for value in np.unique(classes):
                self.groups.append(group[classes == value])
        self.n, self.count, self.device = n, count, device
        self.pointers, self.lengths = pointers, lengths
        self.entries, self.sources = entries, sources
        self.stages = None

    def build(self):
        """Move the tables to the device, each as one tensor."""

        n = self.n
        one, zero = self.count, self.count + 1  # in [coefficients, 1.0, -0.0]
        all_rows, all_x, all_c = [], [], []
        shapes = []
        for rows in self.groups:
            counts = self.lengths[rows]
            height = int(counts.max()) + 1
            x_pick = np.full((height, rows.size + 1), n, dtype=np.int64)
            c_pick = np.full((height, rows.size + 1), zero, dtype=np.int64)
            x_pick[0, :-1] = rows
            c_pick[0, :-1] = one
            # Term number ``rank`` of the row in column ``column`` of the table.
            offsets = np.cumsum(counts) - counts
            rank = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(offsets, counts)
            term = np.repeat(self.pointers[rows], counts) + rank
            column = np.repeat(np.arange(rows.size, dtype=np.int64), counts)
            x_pick[rank + 1, column] = self.sources[term]
            c_pick[rank + 1, column] = self.entries[term]
            all_rows.append(rows)
            all_rows.append(np.array([n + 1], dtype=np.int64))  # the spare column
            all_x.append(x_pick.reshape(-1))
            all_c.append(c_pick.reshape(-1))
            shapes.append((rows.size, height))

        def on_device(parts):
            flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
            return torch.from_numpy(flat).to(self.device)

        self.shapes = shapes
        self.sizes = [height * (rows + 1) for rows, height in shapes]
        self.c_pick = on_device(all_c)
        self.stages = []
        if shapes:
            rows_parts = torch.split(
                on_device(all_rows), [rows + 1 for rows, _ in shapes]
            )
            x_parts = torch.split(on_device(all_x), self.sizes)
            for (rows, height), row_part, x_part in zip(shapes, rows_parts, x_parts):
                self.stages.append((row_part, x_part.view(height, rows + 1)))
        self.pointers = self.lengths = self.entries = self.sources = None

    def bind(self, extended):
        """The coefficient tables, from ``[coefficients, 1.0, -0.0]``."""

        if not self.shapes:
            return []
        parts = torch.split(extended[self.c_pick], self.sizes)
        return [
            part.view(height, rows + 1)
            for (rows, height), part in zip(self.shapes, parts)
        ]

    def run(self, x, coefficients, signed_zero):
        """Sum every stage.  Without ``signed_zero`` a row that C leaves at
        -0.0 becomes +0.0 (see ``torchCSC.sequential_sum``); the caller then
        checks the result for zeros."""

        loop = torch_csc.FORCE_LEVEL_LOOP or x.dtype != torch.float64
        for (rows, pick), coefficient in zip(self.stages, coefficients):
            buffer = torch.take(x, pick)
            buffer *= coefficient
            if loop:
                total = buffer[0].clone()
                for rank in range(1, buffer.shape[0]):
                    total = total + buffer[rank]
            else:
                total = torch_csc.sequential_sum_(buffer, signed_zero)
            x.index_copy_(0, rows, total)


class SolveProgram:
    """The operations of ``LDLSolve`` for the pattern of ``L`` and a permutation."""

    def __init__(self, n, Lp, Li, perm, device):
        self.n = n
        self.perm = torch.tensor(perm, dtype=torch.int64, device=device)
        nnz = Lp[n]
        pointers = np.asarray(Lp, dtype=np.int64)
        rows_of = np.asarray(Li, dtype=np.int64)
        counts = np.diff(pointers)
        column_of = np.repeat(np.arange(n, dtype=np.int64), counts)
        occupied = np.flatnonzero(counts).tolist()

        # QDLDL_Lsolve: x[Li[j]] -= Lx[j] * x[i], i ascending.
        # Step by step: a term is applied once x[i] and the previous term of
        # x[Li[j]] are done.  Row by row: the terms of x[r], which C subtracts
        # in ascending column order, are summed once every x[i] is final.
        # The rows of one column are distinct, so a column is handled at once.
        ready = np.zeros(n, dtype=np.int64)
        level = np.zeros(n, dtype=np.int64)
        steps = np.zeros(nnz, dtype=np.int64)
        for i in occupied:
            first, last = Lp[i], Lp[i + 1]
            rows = rows_of[first:last]
            step = np.maximum(ready[rows], ready[i]) + 1
            ready[rows] = step
            steps[first:last] = step
            level[rows] = np.maximum(level[rows], level[i] + 1)
        # The terms of each row, in ascending column order.
        by_row = np.argsort(rows_of, kind="stable")
        row_pointers = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows_of, minlength=n), out=row_pointers[1:])
        by_rows = _RowSums(
            n, nnz, level, row_pointers, by_row, column_of[by_row], device
        )
        # A step and a group of rows cost about the same, so the form with
        # fewer of them is used.
        self.forward_by_rows = (
            len(by_rows.groups) <= (int(ready.max()) if n else 0)
            if FORWARD_BY_ROWS is None
            else FORWARD_BY_ROWS
        )
        if self.forward_by_rows:
            by_rows.build()
            self.forward = by_rows
        else:
            # step, then target, entry of L, source
            self.forward = _steps(
                steps, (rows_of, np.arange(nnz, dtype=np.int64), column_of), device
            )
            # The entries of L of all steps, so that bind gathers them at once.
            self.forward_sizes = [entry.numel() for _, entry, _ in self.forward]
            self.forward_entry = (
                torch.cat([entry for _, entry, _ in self.forward])
                if self.forward
                else torch.empty(0, dtype=torch.int64, device=device)
            )

        # QDLDL_Ltsolve: x[i] -= Lx[j] * x[Li[j]] over column i, i descending.
        stage = np.zeros(n, dtype=np.int64)
        for i in reversed(occupied):
            stage[i] = 1 + stage[rows_of[Lp[i] : Lp[i + 1]]].max()
        self.backward = _RowSums(
            n, nnz, stage, pointers, np.arange(nnz, dtype=np.int64), rows_of, device
        )
        self.backward.build()

    def bind(self, Lx):
        """Gather ``-Lx`` in the order the steps use it.

        ``a - l * b`` is computed as ``a + (-l) * b``: negation is exact, so the
        product and the sum round as in C.
        """

        negative = -Lx
        extended = torch.cat((negative, Lx.new_tensor([1.0, -0.0])))
        if self.forward_by_rows:
            forward = self.forward.bind(extended)
        else:
            forward = torch.split(negative[self.forward_entry], self.forward_sizes)
        return forward, self.backward.bind(extended), Lx.new_tensor([1.0, 0.0])

    def run(self, bound, Dinv, b):
        # The sums are first done without the correction for sums that C
        # leaves at -0.0.  Such a sum comes out as +0.0, and every row is
        # written once per sweep, so when neither sweep produced a zero no
        # correction was needed.  Otherwise the solve is repeated with it.
        if torch_csc.FORCE_LEVEL_LOOP or b.dtype != torch.float64:
            return self._run(bound, Dinv, b, True)[0]
        result, zero = self._run(bound, Dinv, b, False)
        if bool(zero.item()):
            result = self._run(bound, Dinv, b, True)[0]
        return result

    def _run(self, bound, Dinv, b, signed_zero):
        forward_values, backward_values, tail = bound
        # bp[j] = b[P[j]], followed by the entry that stays 1.0 and the entry
        # for the spare column.
        x = torch.cat((b[self.perm], tail))
        if self.forward_by_rows:
            self.forward.run(x, forward_values, signed_zero)
            zero = None if signed_zero else (x[: self.n] == 0).any()
        else:
            zero = None
            for (target, _, source), negative in zip(self.forward, forward_values):
                # x[Li[j]] -= Lx[j] * x[i]; each target occurs once per step.
                x.index_add_(0, target, negative * torch.take(x, source))
        # x[i] *= Dinv[i]; the two extra entries are multiplied by 1.0.
        x *= torch.cat((Dinv, Dinv.new_ones(2)))
        # val -= Lx[j] * x[Li[j]], j ascending, for every row of a group.
        self.backward.run(x, backward_values, signed_zero)
        # x[P[j]] = bp[j]
        result = torch.empty_like(b)
        result[self.perm] = x[: self.n]
        if not signed_zero:
            zero_result = (result == 0).any()
            zero = zero_result if zero is None else zero | zero_result
        return result, zero
