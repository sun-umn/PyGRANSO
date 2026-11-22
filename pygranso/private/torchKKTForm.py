"""Strict line-by-line Python port of OSQP's KKT assembly and symmetric permute.

Sources:
  osqp/algebra/_common/kkt.c        (form_KKT and its static helpers)
  osqp/algebra/_common/csc_utils.c  (csc_cumsum, csc_pinv, csc_symperm)
  osqp/algebra/_common/lin_sys/qdldl/qdldl_interface.c (permute_KKT)

CSC matrices are dicts ``{"m", "n", "p", "i", "x"}``.  Outputs use plain
Python lists (ints / floats; Python floats are IEEE doubles, so the values
are bit-identical to the C OSQPFloat results).  Control flow and operation
order mirror the C code, including its quirks; nothing is simplified.
"""

from pygranso.private.torchAMD import amd_order


def _ints(values):
    return [int(v) for v in values]


def _floats(values):
    # Keep each scalar's own precision (Python float or numpy.float32).
    return list(values)


def _as_csc(M):
    """Local list view of an input CSC dict (inputs are never modified)."""
    return {
        "m": int(M["m"]),
        "n": int(M["n"]),
        "p": _ints(M["p"]),
        "i": _ints(M["i"]),
        "x": None if M.get("x") is None else _floats(M["x"]),
    }


# --------------------------------------------------------------------------- #
# kkt.c static helpers
# --------------------------------------------------------------------------- #

def _kkt_shifts_param1(KKT, param1, n, format):
    # add an offset to every term in the upper nxn block.
    # assumes triu CSC or CSR format, with fully populated diagonal.
    # format = 0 / CSC:  diagonal terms are last in every column.
    # format = 1 / CSR:  diagonal terms are first in every row.
    offset = 1 if format == 0 else 0
    Kp = KKT["p"]
    Kx = KKT["x"]
    for i in range(n):
        Kx[Kp[i + offset] - offset] += param1


def _kkt_shifts_param2(KKT, param2, param2_sc, startcol, blockwidth, format):
    # *subtract* an offset to every term in the lower mxm block.
    offset = 1 if format == 0 else 0
    Kp = KKT["p"]
    Kx = KKT["x"]
    if param2 is not None:
        for i in range(blockwidth):
            Kx[Kp[i + startcol + offset] - offset] -= param2[i]
    else:
        for i in range(blockwidth):
            Kx[Kp[i + startcol + offset] - offset] -= param2_sc


def _kkt_colcount_diag(K, initcol, blockcols):
    Kp = K["p"]
    for j in range(initcol, initcol + blockcols):
        Kp[j] += 1


def _kkt_colcount_missing_diag(K, M, initcol):
    Kp = K["p"]
    Mp = M["p"]
    Mi = M["i"]
    for j in range(M["n"]):
        # if empty column or last entry not on diagonal..
        if (Mp[j] == Mp[j + 1]) or (Mi[Mp[j + 1] - 1] != j):
            Kp[j + initcol] += 1


def _kkt_colcount_block(K, M, initcol, istranspose):
    Kp = K["p"]
    Mp = M["p"]
    Mi = M["i"]
    if istranspose:
        nnzM = Mp[M["n"]]
        for j in range(nnzM):
            Kp[Mi[j] + initcol] += 1
    else:
        # just add the column count
        for j in range(M["n"]):
            Kp[j + initcol] += Mp[j + 1] - Mp[j]


def _kkt_fill_block(K, M, MtoKKT, initrow, initcol, istranspose):
    Kp = K["p"]
    Ki = K["i"]
    Kx = K["x"]
    Mp = M["p"]
    Mi = M["i"]
    Mx = M["x"]
    for ii in range(M["n"]):
        for jj in range(Mp[ii], Mp[ii + 1]):
            if istranspose:
                col = Mi[jj] + initcol
                row = ii + initrow
            else:
                col = ii + initcol
                row = Mi[jj] + initrow

            dest = Kp[col]
            Kp[col] += 1
            Ki[dest] = row
            Kx[dest] = Mx[jj]
            if MtoKKT is not None:
                MtoKKT[jj] = dest


def _kkt_fill_diag_zeros(K, rhotoKKT, offset, blockdim):
    Kp = K["p"]
    Ki = K["i"]
    Kx = K["x"]
    for j in range(blockdim):
        col = j + offset
        dest = Kp[col]
        Ki[dest] = col
        Kx[dest] = 0.0   # structural zero
        Kp[col] += 1
        if rhotoKKT is not None:
            rhotoKKT[j] = dest


def _kkt_fill_missing_diag_zeros(K, M, offset):
    Kp = K["p"]
    Ki = K["i"]
    Kx = K["x"]
    Mp = M["p"]
    Mi = M["i"]
    for j in range(M["n"]):
        # fill out missing diagonal terms only:
        # if completely empty column last element is not on diagonal..
        if (Mp[j] == Mp[j + 1]) or (Mi[Mp[j + 1] - 1] != j):
            dest = Kp[j + offset]
            Ki[dest] = j + offset
            Kx[dest] = 0.0   # structural zero
            Kp[j] += 1       # (sic) C increments K->p[j], not K->p[j+offset]


def _kkt_colcount_to_colptr(K):
    Kp = K["p"]
    currentptr = 0
    for j in range(K["n"] + 1):
        count = Kp[j]
        Kp[j] = currentptr
        currentptr += count


def _kkt_backshift_colptrs(K):
    Kp = K["p"]
    j = K["n"]
    while j > 0:
        Kp[j] = Kp[j - 1]
        j -= 1
    Kp[0] = 0


def _count_diagonal_entries(P):
    Pp = P["p"]
    Pi = P["i"]
    count = 0
    for j in range(P["n"]):
        # look for nonempty columns with final element
        # on the diagonal.  Assumes triu format.
        if (Pp[j + 1] != Pp[j]) and (Pi[Pp[j + 1] - 1] == j):
            count += 1
    return count


def _kkt_assemble_csr(K, PtoKKT, AtoKKT, rhotoKKT, P, A):
    # NB:  assembling a TRIU KKT in CSR format,
    # which is the same as TRIL KKT in CSC.
    m = A["m"]
    n = P["n"]

    # use K.p to hold nnz entries in each column of the KKT matrix
    for j in range(m + n + 1):
        K["p"][j] = 0
    _kkt_colcount_missing_diag(K, P, 0)
    _kkt_colcount_block(K, P, 0, 1)
    _kkt_colcount_block(K, A, 0, 0)
    _kkt_colcount_diag(K, n, m)

    # cumsum total entries to convert to K.p
    _kkt_colcount_to_colptr(K)

    # fill in value for P, top left (transposed/rowwise)
    _kkt_fill_missing_diag_zeros(K, P, 0)  # before adding P, since tril form
    _kkt_fill_block(K, P, PtoKKT, 0, 0, 1)

    # fill in value for A, lower left (columnwise)
    _kkt_fill_block(K, A, AtoKKT, n, 0, 0)

    # fill in lower right with diagonal of structural zeros
    _kkt_fill_diag_zeros(K, rhotoKKT, n, m)

    # backshift the colptrs to recover K.p again
    _kkt_backshift_colptrs(K)


def _kkt_assemble_csc(K, PtoKKT, AtoKKT, rhotoKKT, P, A):
    m = A["m"]
    n = P["n"]

    # use K.p to hold nnz entries in each column of the KKT matrix
    for j in range(m + n + 1):
        K["p"][j] = 0
    _kkt_colcount_block(K, P, 0, 0)
    _kkt_colcount_missing_diag(K, P, 0)
    _kkt_colcount_block(K, A, n, 1)
    _kkt_colcount_diag(K, n, m)

    # cumsum total entries to convert to K.p
    _kkt_colcount_to_colptr(K)

    # fill in value for P, top left (columnwise)
    _kkt_fill_block(K, P, PtoKKT, 0, 0, 0)
    _kkt_fill_missing_diag_zeros(K, P, 0)  # after adding P, since triu form

    # fill in value for A, lower left (transposed/rowwise)
    _kkt_fill_block(K, A, AtoKKT, 0, n, 1)

    # fill in lower right with diagonal of structural zeros
    _kkt_fill_diag_zeros(K, rhotoKKT, n, m)

    # backshift the colptrs to recover K.p again
    _kkt_backshift_colptrs(K)


# --------------------------------------------------------------------------- #
# form_KKT
# --------------------------------------------------------------------------- #

def form_KKT(P_csc, A_csc, format=0, param1=0.0, param2=None,
             param2_scalar=0.0, return_maps=False):
    """Port of ``form_KKT`` (kkt.c).

    Builds the upper-triangular KKT matrix
        [ P + param1 I        A'        ]
        [      A        -diag(param2)   ]
    ``format == 0``: CSC of the upper triangle.  ``format != 0``: CSR of the
    upper triangle (== CSC of the lower triangle), as in C.

    ``param2`` is a length-m sequence or ``None`` (C ``OSQP_NULL``), in which
    case the scalar ``param2_scalar`` is used.

    Returns the KKT dict ``{"m","n","p","i","x","nzmax"}``.  With
    ``return_maps=True`` returns ``(KKT, PtoKKT, AtoKKT, param2toKKT)``, the
    index maps C writes when the pointers are non-NULL.
    """
    P = _as_csc(P_csc)
    A = _as_csc(A_csc)
    if param2 is not None:
        param2 = _floats(param2)

    # Get matrix dimensions
    m = A["m"]
    n = P["n"]
    nKKT = m + n

    # count elements on diag(P)
    ndiagP = _count_diagonal_entries(P)

    # Get maximum number of nonzero elements (only upper triangular part)
    nnzKKT = (P["p"][n] +   # Number of elements in P
              n -           # Number of elements in param1 * I
              ndiagP +      # remove double count on the diagonal
              A["p"][n] +   # Number of nonzeros in A
              m)            # Number of elements in - diag(param2)

    # Preallocate KKT matrix in csc format
    KKT = {
        "m": nKKT,
        "n": nKKT,
        "nzmax": nnzKKT,
        "p": [0] * (nKKT + 1),
        "i": [0] * nnzKKT,
        "x": [0.0] * nnzKKT,
    }

    if return_maps:
        PtoKKT = [0] * P["p"][n]
        AtoKKT = [0] * A["p"][n]
        rhotoKKT = [0] * m
    else:
        PtoKKT = None
        AtoKKT = None
        rhotoKKT = None

    if format == 0:   # KKT should be built in CSC format
        _kkt_assemble_csc(KKT, PtoKKT, AtoKKT, rhotoKKT, P, A)
    else:             # KKT should be built in CSR format
        _kkt_assemble_csr(KKT, PtoKKT, AtoKKT, rhotoKKT, P, A)

    # apply positive shifts to upper LH diagonal
    _kkt_shifts_param1(KKT, param1, n, format)

    # apply negative shifts to lower RH diagonal
    # NB: rhtoKKT is not needed to do this
    _kkt_shifts_param2(KKT, param2, param2_scalar, n, m, format)

    if not isinstance(param1, float):
        # Structural zeros are written as 0.0; every stored value is exactly
        # representable in the working precision, so restore that type.
        kind = type(param1)
        KKT["x"] = [kind(v) for v in KKT["x"]]

    if return_maps:
        return KKT, PtoKKT, AtoKKT, rhotoKKT
    return KKT


# --------------------------------------------------------------------------- #
# csc_utils.c
# --------------------------------------------------------------------------- #

def csc_cumsum(p, c, n):
    nz = 0
    for i in range(n):
        p[i] = nz
        nz += c[i]
        c[i] = p[i]
    p[n] = nz
    return nz   # return sum (c [0..n-1])


def csc_pinv(perm):
    """Port of ``csc_pinv(p, n)``; ``None`` denotes identity (as in C)."""
    if perm is None:
        return None
    n = len(perm)
    pinv = [0] * n
    for k in range(n):
        pinv[int(perm[k])] = k   # invert the permutation
    return pinv


def csc_symperm(KKT, pinv, AtoC=None, values=1):
    """Port of ``csc_symperm(A, pinv, AtoC, values)``: C = P A P' for an
    upper-triangular symmetric A, with C upper triangular.

    ``AtoC``: ``None`` (C ``OSQP_NULL``) or a list of length nnz(A) that is
    filled in place with the entry map.  Returns the CSC dict C; ``C["x"]``
    is ``None`` when ``values`` is 0 or A has no values.
    """
    n = int(KKT["n"])
    Ap = _ints(KKT["p"])
    Ai = _ints(KKT["i"])
    Ax = None if KKT.get("x") is None else _floats(KKT["x"])
    if pinv is not None:
        pinv = _ints(pinv)

    nzmax = Ap[n]
    Cp = [0] * (n + 1)
    Ci = [0] * nzmax
    Cx = [0.0] * nzmax if (values and (Ax is not None)) else None
    w = [0] * n                             # get workspace (calloc)

    for j in range(n):          # count entries in each column of C
        j2 = pinv[j] if pinv is not None else j   # column j of A is column j2 of C

        for p in range(Ap[j], Ap[j + 1]):
            i = Ai[p]

            if i > j:
                continue        # skip lower triangular part of A
            i2 = pinv[i] if pinv is not None else i   # row i of A is row i2 of C
            w[i2 if i2 > j2 else j2] += 1             # column count of C
    csc_cumsum(Cp, w, n)        # compute column pointers of C

    for j in range(n):
        j2 = pinv[j] if pinv is not None else j   # column j of A is column j2 of C

        for p in range(Ap[j], Ap[j + 1]):
            i = Ai[p]

            if i > j:
                continue        # skip lower triangular part of A
            i2 = pinv[i] if pinv is not None else i   # row i of A is row i2 of C
            col = i2 if i2 > j2 else j2               # c_max(i2, j2)
            q = w[col]
            w[col] += 1
            Ci[q] = i2 if i2 < j2 else j2             # c_min(i2, j2)

            if Cx is not None:
                Cx[q] = Ax[p]

            if AtoC is not None:    # store values of the mappings
                AtoC[p] = q

    return {"m": n, "n": n, "nzmax": nzmax, "p": Cp, "i": Ci, "x": Cx}


# --------------------------------------------------------------------------- #
# qdldl_interface.c: permute_KKT
# --------------------------------------------------------------------------- #

def permute_KKT(KKT, PtoKKT=None, AtoKKT=None, rhotoKKT=None):
    """Port of ``permute_KKT`` (qdldl_interface.c).

    Runs AMD on the upper-triangular KKT, inverts the permutation and applies
    ``csc_symperm``.  The maps that are not ``None`` are updated in place to
    index the permuted matrix (``map[i] = KtoPKPt[map[i]]``).

    Returns ``(amd_status, perm, KKT_permuted)``; on a negative AMD status
    returns ``(amd_status, None, None)`` like the C early return.
    """
    amd_status, perm = amd_order(int(KKT["n"]), KKT["p"], KKT["i"])
    if amd_status < 0:
        return amd_status, None, None

    # Inverse of the permutation vector
    Pinv = csc_pinv(perm)

    # Permute KKT matrix
    if PtoKKT is None and AtoKKT is None and rhotoKKT is None:
        KKT_temp = csc_symperm(KKT, Pinv, None, 1)
    else:
        # Allocate vector of mappings from unpermuted to permuted
        KtoPKPt = [0] * int(KKT["p"][int(KKT["n"])])
        KKT_temp = csc_symperm(KKT, Pinv, KtoPKPt, 1)

        # Update vectors PtoKKT, AtoKKT and rhotoKKT
        if PtoKKT is not None:
            for i in range(len(PtoKKT)):
                PtoKKT[i] = KtoPKPt[PtoKKT[i]]
        if AtoKKT is not None:
            for i in range(len(AtoKKT)):
                AtoKKT[i] = KtoPKPt[AtoKKT[i]]
        if rhotoKKT is not None:
            for i in range(len(rhotoKKT)):
                rhotoKKT[i] = KtoPKPt[rhotoKKT[i]]

    return amd_status, perm, KKT_temp
