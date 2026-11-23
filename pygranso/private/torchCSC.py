"""CSC storage and the ``csc_math.c`` operations used by the Torch OSQP solver.

OSQP keeps ``P`` (upper triangle) and ``A`` in compressed sparse column form.
The pattern is fixed by ``osqp_setup``; every later operation visits the stored
entries only, in storage order.  The classes here keep the same data: a pattern
(row and column index of every stored entry, column-major, rows ascending) and
one value per stored entry, explicit zeros included.
"""

from __future__ import annotations

import torch

# Test hook: use the per-rank loop even where the padded cumsum is available.
FORCE_LEVEL_LOOP = False


class _Plan:
    """Accumulation order of one product: ``out[k] += value[k] * x[inp[k]]``.

    Entries are sorted by output index, then by input index.  That is the order
    in which ``csc_Axpy``, ``csc_Atxpy`` and ``csc_Axpy_sym_triu`` add terms to
    one output entry, so adding the terms of each output in this order gives
    the same rounded sum as C.
    """

    def __init__(self, out, inp, val, n_out, n_in):
        order = torch.argsort(out * max(int(n_in), 1) + inp, stable=True)
        self.out, self.inp, self.val = out[order], inp[order], val[order]
        self.n_out, self.n_in = int(n_out), int(n_in)
        # Row-major, columns ascending: the index order of a coalesced COO tensor.
        self.indices = torch.stack((self.out, self.inp))
        counts = torch.bincount(self.out, minlength=self.n_out)
        starts = torch.cumsum(counts, 0) - counts
        self.rank = (
            torch.arange(self.out.numel(), device=out.device) - starts[self.out]
        )
        self.width = int(counts.max().item()) if self.out.numel() else 0
        self._ell = None
        self._levels = None

    def accumulate(self, products, start):
        """Sequential ``acc += term`` per output, from ``+0`` or from ``start``."""

        if self.width == 0:
            if start is not None:
                return start.clone()
            return products.new_zeros(self.n_out)
        padded = self.n_out * (self.width + 1)
        if (
            not FORCE_LEVEL_LOOP
            and products.device.type == "cpu"
            and products.dtype == torch.float64
            and padded <= 4 * products.numel() + 4096
        ):
            # torch.cumsum is the recurrence acc += term for CPU float64.  Rows
            # shorter than the widest one are padded with -0.0, whose addition
            # leaves every value unchanged, signed zeros included.
            if self._ell is None:
                self._ell = self.out * (self.width + 1) + self.rank + 1
            buffer = products.new_full((self.n_out, self.width + 1), -0.0)
            buffer[:, 0] = 0.0 if start is None else start
            buffer.view(-1)[self._ell] = products
            return torch.cumsum(buffer, dim=1)[:, -1]
        if self._levels is None:
            order = torch.argsort(self.rank, stable=True)
            sizes = torch.bincount(self.rank, minlength=self.width).tolist()
            self._levels = [
                (chunk, self.out[chunk]) for chunk in torch.split(order, sizes)
            ]
        total = products.new_zeros(self.n_out) if start is None else start.clone()
        for chunk, rows in self._levels:
            # Each output index occurs at most once per level.
            total[rows] = total[rows] + products[chunk]
        return total


class CSCPattern:
    """Row and column index of every stored entry, column-major, rows ascending."""

    def __init__(self, m, n, rows, cols):
        self.m, self.n = int(m), int(n)
        self.rows, self.cols = rows, cols
        self._cache = {}

    @property
    def nnz(self):
        return int(self.rows.numel())

    def same(self, other):
        if self is other:
            return True
        return (
            self.m == other.m
            and self.n == other.n
            and self.nnz == other.nnz
            and self.rows.device == other.rows.device
            and torch.equal(self.rows, other.rows)
            and torch.equal(self.cols, other.cols)
        )

    def is_upper_triangular(self):
        return not bool(torch.any(self.rows > self.cols).item())

    def lists(self):
        """Column pointers and row indices as Python lists (``Ap``, ``Ai``)."""

        if "lists" not in self._cache:
            counts = torch.bincount(self.cols, minlength=self.n)
            pointers = [0] + torch.cumsum(counts, 0).tolist()
            self._cache["lists"] = (pointers, self.rows.tolist())
        return self._cache["lists"]

    def plan(self, kind):
        if kind not in self._cache:
            index = torch.arange(self.nnz, device=self.rows.device)
            if kind == "Ax":
                plan = _Plan(self.rows, self.cols, index, self.m, self.n)
            elif kind == "Atx":
                plan = _Plan(self.cols, self.rows, index, self.n, self.m)
            else:
                # csc_Axpy_sym_triu: entry (i, j) adds to y[i] with x[j] and,
                # off the diagonal, to y[j] with x[i].
                off = self.rows != self.cols
                plan = _Plan(
                    torch.cat((self.rows, self.cols[off])),
                    torch.cat((self.cols, self.rows[off])),
                    torch.cat((index, index[off])),
                    self.n,
                    self.n,
                )
            self._cache[kind] = plan
        return self._cache[kind]


class CSCMatrix:
    """A pattern with one value per stored entry.  Instances are not modified.

    ``symmetric`` marks ``P``: only the upper triangle is stored and products
    use the symmetric matrix, as ``OSQPMatrix`` does with ``symmetry == TRIU``.
    """

    def __init__(self, pattern, values, symmetric=False):
        self.pattern = pattern
        self.values = values
        self.symmetric = bool(symmetric)

    @classmethod
    def from_dense(cls, matrix, *, symmetric=False, mask=None):
        """Stored entries are the nonzeros (of the upper triangle for ``P``)."""

        mask = matrix.ne(0) if mask is None else mask
        if symmetric:
            mask = torch.triu(mask)
        index = mask.T.nonzero()
        cols, rows = index[:, 0], index[:, 1]
        pattern = CSCPattern(matrix.shape[0], matrix.shape[1], rows, cols)
        return cls(pattern, matrix[rows, cols], symmetric)

    @classmethod
    def from_sparse(cls, tensor):
        """Keep every stored entry of a COO, CSR or CSC tensor, zeros included."""

        if tensor.ndim != 2:
            raise ValueError("A sparse matrix input must be two-dimensional.")
        m, n = tensor.shape
        device = tensor.device
        if tensor.layout == torch.sparse_coo:
            tensor = tensor.coalesce()
            rows, cols = tensor.indices()
            values = tensor.values()
        elif tensor.layout == torch.sparse_csc:
            counts = torch.diff(tensor.ccol_indices())
            cols = torch.repeat_interleave(torch.arange(n, device=device), counts)
            rows, values = tensor.row_indices(), tensor.values()
        elif tensor.layout == torch.sparse_csr:
            counts = torch.diff(tensor.crow_indices())
            rows = torch.repeat_interleave(torch.arange(m, device=device), counts)
            cols, values = tensor.col_indices(), tensor.values()
        else:
            raise ValueError(f"Unsupported sparse layout {tensor.layout}.")
        if values.ndim != 1:
            raise ValueError("Sparse matrix values must be scalars.")
        rows, cols = rows.to(torch.int64), cols.to(torch.int64)
        key = cols * max(int(m), 1) + rows
        order = torch.argsort(key, stable=True)
        key = key[order]
        if bool(torch.any(key[1:] == key[:-1]).item()):
            raise ValueError("A sparse matrix input stores an entry more than once.")
        return cls(CSCPattern(m, n, rows[order], cols[order]), values[order])

    @property
    def shape(self):
        return (self.pattern.m, self.pattern.n)

    @property
    def device(self):
        return self.values.device

    @property
    def dtype(self):
        return self.values.dtype

    def with_values(self, values):
        return CSCMatrix(self.pattern, values, self.symmetric)

    def equal(self, other):
        return (
            isinstance(other, CSCMatrix)
            and self.symmetric == other.symmetric
            and self.device == other.device
            and self.dtype == other.dtype
            and self.pattern.same(other.pattern)
            and torch.equal(self.values, other.values)
        )

    def to_dense(self):
        dense = self.values.new_zeros(self.shape)
        rows, cols = self.pattern.rows, self.pattern.cols
        dense[rows, cols] = self.values
        if self.symmetric:
            dense[cols, rows] = self.values
        return dense

    def fit_dense(self, matrix):
        """Values of ``matrix`` on this pattern, or ``None`` if it has a nonzero elsewhere.

        ``osqp_update_data_mat`` replaces stored values only.  A stored entry
        that becomes zero stays stored; a nonzero outside the pattern cannot be
        expressed as an update.
        """

        if tuple(matrix.shape) != self.shape:
            return None
        values = matrix[self.pattern.rows, self.pattern.cols]
        source = torch.triu(matrix) if self.symmetric else matrix
        if int(torch.count_nonzero(source)) != int(torch.count_nonzero(values)):
            return None
        return self.with_values(values)

    # csc_Axpy / csc_Axpy_sym_triu / csc_Atxpy with alpha = +-1.

    def matvec(self, x, start=None, subtract=False):
        plan = self.pattern.plan("sym" if self.symmetric else "Ax")
        return self._product(plan, x, start, subtract)

    def rmatvec(self, y, start=None, subtract=False):
        return self._product(self.pattern.plan("Atx"), y, start, subtract)

    def _product(self, plan, vector, start, subtract):
        values = self.values[plan.val]
        if (
            start is None
            and not subtract
            and not FORCE_LEVEL_LOOP
            and plan.width > 0
            and values.device.type == "cpu"
            and values.dtype == torch.float64
        ):
            # torch.sparse.mm on a coalesced COO tensor adds value * x[column]
            # to its row entry by entry in index order, which is the C order.
            # The CSR and CSC kernels and every CUDA kernel use another order.
            matrix = torch.sparse_coo_tensor(
                plan.indices, values, (plan.n_out, plan.n_in), is_coalesced=True
            )
            return torch.sparse.mm(matrix, vector.unsqueeze(1)).squeeze(1)
        products = values * vector[plan.inp]
        return plan.accumulate(-products if subtract else products, start)

    def to_sparse_csc(self):
        """The stored entries as a ``torch.sparse_csc_tensor`` (``Ap``, ``Ai``, ``Ax``)."""

        pointers, _ = self.pattern.lists()
        return torch.sparse_csc_tensor(
            torch.tensor(pointers, dtype=torch.int64, device=self.device),
            self.pattern.rows,
            self.values,
            size=self.shape,
        )

    # csc_col_norm_inf / csc_row_norm_inf: maxima of |x| from a +0 start.

    def col_norm_inf(self):
        result = self.values.new_zeros(self.pattern.n)
        return result.scatter_reduce(
            0, self.pattern.cols, torch.abs(self.values), "amax", include_self=True
        )

    def row_norm_inf(self):
        result = self.values.new_zeros(self.pattern.m)
        return result.scatter_reduce(
            0, self.pattern.rows, torch.abs(self.values), "amax", include_self=True
        )

    def scale(self, left, right):
        """``csc_lmult_diag`` then ``csc_rmult_diag``: ``(left[i] * x) * right[j]``."""

        rows, cols = self.pattern.rows, self.pattern.cols
        return self.with_values((left[rows] * self.values) * right[cols])

    def times(self, scalar):
        return self.with_values(scalar * self.values)

    def select_rows(self, indices):
        """Rows ``indices`` (ascending) in their order, stored entries kept."""

        pattern = self.pattern
        new_row = torch.full((pattern.m,), -1, dtype=torch.int64, device=self.device)
        new_row[indices] = torch.arange(indices.numel(), device=self.device)
        mapped = new_row[pattern.rows]
        keep = mapped >= 0
        reduced = CSCPattern(indices.numel(), pattern.n, mapped[keep], pattern.cols[keep])
        return CSCMatrix(reduced, self.values[keep], self.symmetric)

    def row_hashes(self):
        """One integer per row that ignores stored zeros (used to compare rows)."""

        values = self.values + 0.0
        bits = values.view(torch.int64 if values.dtype == torch.float64 else torch.int32)
        bits = bits.to(torch.int64)
        mixed = (bits ^ (bits >> 29)) * ((self.pattern.cols + 1) * -7046029254386353131)
        result = torch.zeros(self.pattern.m, dtype=torch.int64, device=self.device)
        return result.index_add_(0, self.pattern.rows, mixed)


class KKTSystem:
    """``[P + sigma I, A'; A, -diag(rho_inv)]`` described by its parts.

    ``rho_inv_vec`` is ``None`` for the polishing system, where C passes the
    scalar ``sigma`` for the lower block.
    """

    def __init__(self, P, A, sigma, rho_inv_vec):
        self.P, self.A = P, A
        self.sigma = float(sigma)
        self.rho_inv_vec = rho_inv_vec

    @property
    def shape(self):
        size = self.P.shape[0] + self.A.shape[0]
        return (size, size)

    @property
    def device(self):
        return self.P.device

    @property
    def dtype(self):
        return self.P.dtype

    def top_equal(self, other):
        """Same ``[P + sigma I, A']`` rows: only the rho block may differ."""

        return (
            isinstance(other, KKTSystem)
            and self.sigma == other.sigma
            and self.P.equal(other.P)
            and self.A.equal(other.A)
        )

    def equal(self, other):
        if not self.top_equal(other):
            return False
        if self.rho_inv_vec is None or other.rho_inv_vec is None:
            return self.rho_inv_vec is None and other.rho_inv_vec is None
        return torch.equal(self.rho_inv_vec, other.rho_inv_vec)

    def matvec(self, vector):
        n = self.P.shape[0]
        x, nu = vector[:n], vector[n:]
        lower = self.sigma * nu if self.rho_inv_vec is None else self.rho_inv_vec * nu
        return torch.cat(
            (
                self.P.matvec(x) + self.sigma * x + self.A.rmatvec(nu),
                self.A.matvec(x) - lower,
            )
        )

    def to_dense(self):
        n, m = self.P.shape[0], self.A.shape[0]
        A = self.A.to_dense()
        identity = torch.eye(n, device=self.device, dtype=self.dtype)
        lower = (
            self.sigma * torch.ones(m, device=self.device, dtype=self.dtype)
            if self.rho_inv_vec is None
            else self.rho_inv_vec
        )
        top = torch.cat((self.P.to_dense() + self.sigma * identity, A.T), dim=1)
        return torch.cat((top, torch.cat((A, -torch.diag(lower)), dim=1)), dim=0)
