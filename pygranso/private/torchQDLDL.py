"""Line-by-line translation of the pinned QDLDL routines used by OSQP.

Source: ``qdldl.c`` at QDLDL commit 138fdac58b9cd1c4137ff1b99152c8108a6cff5b
(``QDLDL_etree``, ``QDLDL_factor``, ``QDLDL_Lsolve``, ``QDLDL_Ltsolve``,
``QDLDL_solve``) and ``LDLSolve`` in OSQP's ``qdldl_interface.c``.

The factorization is a sequence of scalar IEEE operations whose order fixes the
rounding.  The loops below keep that order: one multiplication followed by one
subtraction per update, in the same index sequence as the C code.  Values are
Python floats (binary64) or NumPy float32 scalars, so every operation rounds
exactly as the corresponding ``QDLDL_float`` operation does.
"""

from __future__ import annotations

QDLDL_UNKNOWN = -1
QDLDL_USED = 1
QDLDL_UNUSED = 0
QDLDL_INT_MAX = 2**63 - 1


def qdldl_etree(n, Ap, Ai):
    """``QDLDL_etree``: returns ``(sumLnz, Lnz, etree)``."""

    work = [0] * n
    Lnz = [0] * n
    etree = [QDLDL_UNKNOWN] * n

    for i in range(n):
        # Abort if A doesn't have at least one entry in every column
        if Ap[i] == Ap[i + 1]:
            return -1, Lnz, etree

    for j in range(n):
        work[j] = j
        for p in range(Ap[j], Ap[j + 1]):
            i = Ai[p]
            # Abort if entries on lower triangle
            if i > j:
                return -1, Lnz, etree
            while work[i] != j:
                if etree[i] == QDLDL_UNKNOWN:
                    etree[i] = j
                Lnz[i] += 1
                work[i] = j
                i = etree[i]

    sumLnz = 0
    for i in range(n):
        if sumLnz > QDLDL_INT_MAX - Lnz[i]:
            sumLnz = -2
            break
        sumLnz += Lnz[i]
    return sumLnz, Lnz, etree


def qdldl_factor(n, Ap, Ai, Ax, Lnz, etree, sumLnz, zero=0.0, one=1.0):
    """``QDLDL_factor``: returns ``(positiveValuesInD or -1, Lp, Li, Lx, D, Dinv)``.

    ``zero`` and ``one`` carry the scalar type (Python float or ``numpy.float32``).
    """

    Lp = [0] * (n + 1)
    Li = [0] * sumLnz
    Lx = [zero] * sumLnz
    D = [zero] * n
    Dinv = [zero] * n
    yMarkers = [QDLDL_UNUSED] * n
    yIdx = [0] * n
    elimBuffer = [0] * n
    LNextSpaceInCol = [0] * n
    yVals = [zero] * n
    positiveValuesInD = 0

    for i in range(n):
        Lp[i + 1] = Lp[i] + Lnz[i]
        LNextSpaceInCol[i] = Lp[i]

    # First element of the diagonal D.
    D[0] = Ax[0]
    if D[0] == 0.0:
        return -1, Lp, Li, Lx, D, Dinv
    if D[0] > 0.0:
        positiveValuesInD += 1
    Dinv[0] = one / D[0]

    for k in range(1, n):
        nnzY = 0
        tmpIdx = Ap[k + 1]
        for i in range(Ap[k], tmpIdx):
            bidx = Ai[i]
            if bidx == k:
                D[k] = Ax[i]
                continue
            yVals[bidx] = Ax[i]
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
                # Put the buffered elimination list into the ordering in reverse
                while nnzE:
                    nnzE -= 1
                    yIdx[nnzY] = elimBuffer[nnzE]
                    nnzY += 1

        for i in range(nnzY - 1, -1, -1):
            cidx = yIdx[i]
            tmpIdx = LNextSpaceInCol[cidx]
            yVals_cidx = yVals[cidx]
            for j in range(Lp[cidx], tmpIdx):
                yVals[Li[j]] -= Lx[j] * yVals_cidx
            Li[tmpIdx] = k
            Lx[tmpIdx] = yVals_cidx * Dinv[cidx]
            D[k] -= yVals_cidx * Lx[tmpIdx]
            LNextSpaceInCol[cidx] += 1
            yVals[cidx] = zero
            yMarkers[cidx] = QDLDL_UNUSED

        if D[k] == 0.0:
            return -1, Lp, Li, Lx, D, Dinv
        if D[k] > 0.0:
            positiveValuesInD += 1
        Dinv[k] = one / D[k]

    return positiveValuesInD, Lp, Li, Lx, D, Dinv


def qdldl_lsolve(n, Lp, Li, Lx, x):
    """``QDLDL_Lsolve``: solves ``(L + I) x = b`` in place."""

    for i in range(n):
        val = x[i]
        for j in range(Lp[i], Lp[i + 1]):
            x[Li[j]] -= Lx[j] * val


def qdldl_ltsolve(n, Lp, Li, Lx, x):
    """``QDLDL_Ltsolve``: solves ``(L + I)' x = b`` in place."""

    for i in range(n - 1, -1, -1):
        val = x[i]
        for j in range(Lp[i], Lp[i + 1]):
            val -= Lx[j] * x[Li[j]]
        x[i] = val


def qdldl_solve(n, Lp, Li, Lx, Dinv, x):
    """``QDLDL_solve``: solves ``L D L' x = b`` in place."""

    qdldl_lsolve(n, Lp, Li, Lx, x)
    for i in range(n):
        x[i] *= Dinv[i]
    qdldl_ltsolve(n, Lp, Li, Lx, x)


def ldl_solve(b, Lp, Li, Lx, Dinv, perm):
    """``LDLSolve`` in ``qdldl_interface.c``: solves ``P' L D L' P x = b``."""

    n = len(perm)
    bp = [b[perm[j]] for j in range(n)]
    qdldl_solve(n, Lp, Li, Lx, Dinv, bp)
    x = [None] * n
    for j in range(n):
        x[perm[j]] = bp[j]
    return x
