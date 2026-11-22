"""Strict line-by-line Python port of the AMD ordering bundled with OSQP/QDLDL.

Source: osqp/algebra/_common/lin_sys/qdldl/amd/src/{amd_order, amd_valid,
amd_preprocess, amd_aat, amd_1, amd_2, amd_postorder, amd_post_tree}.c
(AMD, Copyright (c) Timothy A. Davis, Patrick R. Amestoy, and Iain S. Duff).

This mirrors the long-integer build (``amd_l_order``) called with
``Control == NULL`` (dense = 10, aggressive = 1).  Control flow and the order
of operations follow the C code; nothing is simplified.  The ``Info``
statistics (doubles) are not computed: in the C code they are write-only and
never feed back into any integer decision.

All workspaces are plain Python lists of ints.  The C code carves the six
size-n workspaces and ``Iw`` out of one array ``S``; here they are separate
lists, which is equivalent because the C code never indexes across them.
"""

import math

AMD_OK = 0
AMD_OUT_OF_MEMORY = -1
AMD_INVALID = -2
AMD_OK_BUT_JUMBLED = 1

AMD_DEFAULT_DENSE = 10.0
AMD_DEFAULT_AGGRESSIVE = 1

EMPTY = -1

# Int is SuiteSparse_long (64-bit "long long"), but Int_MAX is
# SuiteSparse_long_max == LONG_MAX (SuiteSparse_config.h), which is 2**31 - 1
# for the Windows build of the pinned C library (LONG_MAX is 2**63 - 1 on
# LP64 platforms).  It only enters through wbig in clear_flag and the slen
# size check.
AMD_INT_MAX = 2147483647
# size_t / unsigned Int are 64-bit.
_SIZE_T_MAX = (1 << 64) - 1
_SIZEOF_INT = 8
_UINT_MASK = (1 << 64) - 1


def _flip(i):
    return -i - 2


# ========================================================================= #
# === AMD_valid =========================================================== #
# ========================================================================= #

def amd_valid(n_row, n_col, Ap, Ai):
    result = AMD_OK

    if n_row < 0 or n_col < 0 or Ap is None or Ai is None:
        return AMD_INVALID
    nz = Ap[n_col]
    if Ap[0] != 0 or nz < 0:
        # column pointers must start at Ap [0] = 0, and Ap [n] must be >= 0
        return AMD_INVALID
    for j in range(n_col):
        p1 = Ap[j]
        p2 = Ap[j + 1]
        if p1 > p2:
            # column pointers must be ascending
            return AMD_INVALID
        ilast = EMPTY
        for p in range(p1, p2):
            i = Ai[p]
            if i < 0 or i >= n_row:
                # row index out of range
                return AMD_INVALID
            if i <= ilast:
                # row index unsorted, or duplicate entry present
                result = AMD_OK_BUT_JUMBLED
            ilast = i
    return result


# ========================================================================= #
# === AMD_preprocess ====================================================== #
# ========================================================================= #

def amd_preprocess(n, Ap, Ai, Rp, Ri, W, Flag):
    # count the entries in each row of A (excluding duplicates)
    for i in range(n):
        W[i] = 0            # # of nonzeros in row i (excl duplicates)
        Flag[i] = EMPTY     # Flag [i] = j if i appears in column j
    for j in range(n):
        p2 = Ap[j + 1]
        for p in range(Ap[j], p2):
            i = Ai[p]
            if Flag[i] != j:
                # row index i has not yet appeared in column j
                W[i] += 1       # one more entry in row i
                Flag[i] = j     # flag row index i as appearing in col j

    # compute the row pointers for R
    Rp[0] = 0
    for i in range(n):
        Rp[i + 1] = Rp[i] + W[i]
    for i in range(n):
        W[i] = Rp[i]
        Flag[i] = EMPTY

    # construct the row form matrix R
    for j in range(n):
        p2 = Ap[j + 1]
        for p in range(Ap[j], p2):
            i = Ai[p]
            if Flag[i] != j:
                # row index i has not yet appeared in column j
                Ri[W[i]] = j    # put col j in row i
                W[i] += 1
                Flag[i] = j     # flag row index i as appearing in col j


# ========================================================================= #
# === AMD_aat ============================================================= #
# ========================================================================= #

def amd_aat(n, Ap, Ai, Len, Tp):
    """Returns nz in A+A' (excluding the diagonal).  Info is not computed."""
    for k in range(n):
        Len[k] = 0

    nzdiag = 0
    nzboth = 0
    nz = Ap[n]  # noqa: F841  (Info only)

    for k in range(n):
        p1 = Ap[k]
        p2 = Ap[k + 1]

        # construct A+A'
        p = p1
        while p < p2:
            # scan the upper triangular part of A
            j = Ai[p]
            if j < k:
                # entry A (j,k) is in the strictly upper triangular part,
                # add both A (j,k) and A (k,j) to the matrix A+A'
                Len[j] += 1
                Len[k] += 1
                p += 1
            elif j == k:
                # skip the diagonal
                p += 1
                nzdiag += 1
                break
            else:  # j > k
                # first entry below the diagonal
                break
            # scan lower triangular part of A, in column j until reaching
            # row k.  Start where last scan left off.
            pj2 = Ap[j + 1]
            pj = Tp[j]
            while pj < pj2:
                i = Ai[pj]
                if i < k:
                    # A (i,j) is only in the lower part, not in upper.
                    # add both A (i,j) and A (j,i) to the matrix A+A'
                    Len[i] += 1
                    Len[j] += 1
                    pj += 1
                elif i == k:
                    # entry A (k,j) in lower part and A (j,k) in upper
                    pj += 1
                    nzboth += 1
                    break
                else:  # i > k
                    # consider this entry later, when k advances to i
                    break
            Tp[j] = pj
        # Tp [k] points to the entry just below the diagonal in column k
        Tp[k] = p

    # clean up, for remaining mismatched entries
    for j in range(n):
        for pj in range(Tp[j], Ap[j + 1]):
            i = Ai[pj]
            # A (i,j) is only in the lower part, not in upper.
            # add both A (i,j) and A (j,i) to the matrix A+A'
            Len[i] += 1
            Len[j] += 1

    # (symmetry statistic: Info only, skipped)

    nzaat = 0
    for k in range(n):
        nzaat += Len[k]
    return nzaat


# ========================================================================= #
# === AMD_1 =============================================================== #
# ========================================================================= #

def amd_1(n, Ap, Ai, P, Pinv, Len, slen, stats=None):
    iwlen = slen - 6 * n
    Pe = [0] * n
    Nv = [0] * n
    Head = [0] * n
    Elen = [0] * n
    Degree = [0] * n
    W = [0] * n
    Iw = [0] * iwlen

    # construct the pointers for A+A'
    Sp = Nv     # use Nv and W as workspace for Sp and Tp [
    Tp = W
    pfree = 0
    for j in range(n):
        Pe[j] = pfree
        Sp[j] = pfree
        pfree += Len[j]

    for k in range(n):
        p1 = Ap[k]
        p2 = Ap[k + 1]

        # construct A+A'
        p = p1
        while p < p2:
            # scan the upper triangular part of A
            j = Ai[p]
            if j < k:
                # entry A (j,k) in the strictly upper triangular part
                Iw[Sp[j]] = k
                Sp[j] += 1
                Iw[Sp[k]] = j
                Sp[k] += 1
                p += 1
            elif j == k:
                # skip the diagonal
                p += 1
                break
            else:  # j > k
                # first entry below the diagonal
                break
            # scan lower triangular part of A, in column j until reaching
            # row k.  Start where last scan left off.
            pj2 = Ap[j + 1]
            pj = Tp[j]
            while pj < pj2:
                i = Ai[pj]
                if i < k:
                    # A (i,j) is only in the lower part, not in upper
                    Iw[Sp[i]] = j
                    Sp[i] += 1
                    Iw[Sp[j]] = i
                    Sp[j] += 1
                    pj += 1
                elif i == k:
                    # entry A (k,j) in lower part and A (j,k) in upper
                    pj += 1
                    break
                else:  # i > k
                    # consider this entry later, when k advances to i
                    break
            Tp[j] = pj
        Tp[k] = p

    # clean up, for remaining mismatched entries
    for j in range(n):
        for pj in range(Tp[j], Ap[j + 1]):
            i = Ai[pj]
            # A (i,j) is only in the lower part, not in upper
            Iw[Sp[i]] = j
            Sp[i] += 1
            Iw[Sp[j]] = i
            Sp[j] += 1

    # Tp and Sp no longer needed ]

    # order the matrix
    amd_2(n, Pe, Iw, Len, iwlen, pfree,
          Nv, Pinv, P, Head, Elen, Degree, W, stats=stats)


# ========================================================================= #
# === clear_flag ========================================================== #
# ========================================================================= #

def _clear_flag(wflg, wbig, W, n):
    if wflg < 2 or wflg >= wbig:
        for x in range(n):
            if W[x] != 0:
                W[x] = 1
        wflg = 2
    # at this point, W [0..n-1] < wflg holds
    return wflg


# ========================================================================= #
# === AMD_2 =============================================================== #
# ========================================================================= #

def amd_2(n, Pe, Iw, Len, iwlen, pfree,
          Nv, Next, Last, Head, Elen, Degree, W,
          alpha=AMD_DEFAULT_DENSE, aggressive=AMD_DEFAULT_AGGRESSIVE,
          stats=None):
    # ===================================================================== #
    #  INITIALIZATIONS
    # ===================================================================== #

    me = EMPTY
    mindeg = 0
    ncmpa = 0
    nel = 0
    lemax = 0

    # get control parameters (Control == NULL: defaults passed as arguments)
    if alpha < 0:
        # only remove completely dense rows/columns
        dense = n - 2
    else:
        dense = int(alpha * math.sqrt(float(n)))
    dense = max(16, dense)
    dense = min(n, dense)

    for i in range(n):
        Last[i] = EMPTY
        Head[i] = EMPTY
        Next[i] = EMPTY
        Nv[i] = 1
        W[i] = 1
        Elen[i] = 0
        Degree[i] = Len[i]

    # initialize wflg
    wbig = AMD_INT_MAX - n
    wflg = _clear_flag(0, wbig, W, n)

    # --------------------------------------------------------------------- #
    # initialize degree lists and eliminate dense and empty rows
    # --------------------------------------------------------------------- #

    ndense = 0

    for i in range(n):
        deg = Degree[i]
        if deg == 0:
            # we have a variable that can be eliminated at once because
            # there is no off-diagonal non-zero in its row.
            Elen[i] = _flip(1)
            nel += 1
            Pe[i] = EMPTY
            W[i] = 0
        elif deg > dense:
            # Dense variables are not treated as elements, but as unordered,
            # non-principal variables that have no parent.
            ndense += 1
            Nv[i] = 0       # do not postorder this node
            Elen[i] = EMPTY
            nel += 1
            Pe[i] = EMPTY
        else:
            # place i in the degree list corresponding to its degree
            inext = Head[deg]
            if inext != EMPTY:
                Last[inext] = i
            Next[i] = inext
            Head[deg] = i

    # ===================================================================== #
    # WHILE (selecting pivots) DO
    # ===================================================================== #

    while nel < n:

        # ================================================================= #
        # GET PIVOT OF MINIMUM DEGREE
        # ================================================================= #

        # find next supervariable for elimination
        deg = mindeg
        while deg < n:
            me = Head[deg]
            if me != EMPTY:
                break
            deg += 1
        mindeg = deg

        # remove chosen variable from link list
        inext = Next[me]
        if inext != EMPTY:
            Last[inext] = EMPTY
        Head[deg] = inext

        # me represents the elimination of pivots nel to nel+Nv[me]-1.
        # place me itself as the first in this set.
        elenme = Elen[me]
        nvpiv = Nv[me]
        nel += nvpiv

        # ================================================================= #
        # CONSTRUCT NEW ELEMENT
        # ================================================================= #

        # flag the variable "me" as being in Lme by negating Nv [me]
        Nv[me] = -nvpiv
        degme = 0

        if elenme == 0:
            # ------------------------------------------------------------- #
            # construct the new element in place
            # ------------------------------------------------------------- #
            pme1 = Pe[me]
            pme2 = pme1 - 1

            p = pme1
            while p <= pme1 + Len[me] - 1:
                i = Iw[p]
                nvi = Nv[i]
                if nvi > 0:
                    # i is a principal variable not yet placed in Lme.
                    # store i in new list

                    # flag i as being in Lme by negating Nv [i]
                    degme += nvi
                    Nv[i] = -nvi
                    pme2 += 1
                    Iw[pme2] = i

                    # remove variable i from degree list.
                    ilast = Last[i]
                    inext = Next[i]
                    if inext != EMPTY:
                        Last[inext] = ilast
                    if ilast != EMPTY:
                        Next[ilast] = inext
                    else:
                        # i is at the head of the degree list
                        Head[Degree[i]] = inext
                p += 1
        else:
            # ------------------------------------------------------------- #
            # construct the new element in empty space, Iw [pfree ...]
            # ------------------------------------------------------------- #
            p = Pe[me]
            pme1 = pfree
            slenme = Len[me] - elenme

            knt1 = 1
            while knt1 <= elenme + 1:
                if knt1 > elenme:
                    # search the supervariables in me.
                    e = me
                    pj = p
                    ln = slenme
                else:
                    # search the elements in me.
                    e = Iw[p]
                    p += 1
                    pj = Pe[e]
                    ln = Len[e]

                # search for different supervariables and add them to the
                # new list, compressing when necessary.
                knt2 = 1
                while knt2 <= ln:
                    i = Iw[pj]
                    pj += 1
                    nvi = Nv[i]
                    if nvi > 0:
                        # ------------------------------------------------- #
                        # compress Iw, if necessary
                        # ------------------------------------------------- #
                        if pfree >= iwlen:
                            # prepare for compressing Iw by adjusting pointers
                            # and lengths so that the lists being searched in
                            # the inner and outer loops contain only the
                            # remaining entries.
                            Pe[me] = p
                            Len[me] -= knt1
                            # check if nothing left of supervariable me
                            if Len[me] == 0:
                                Pe[me] = EMPTY
                            Pe[e] = pj
                            Len[e] = ln - knt2
                            # nothing left of element e
                            if Len[e] == 0:
                                Pe[e] = EMPTY

                            ncmpa += 1      # one more garbage collection

                            # store first entry of each object in Pe
                            # FLIP the first entry in each object
                            for j in range(n):
                                pn = Pe[j]
                                if pn >= 0:
                                    Pe[j] = Iw[pn]
                                    Iw[pn] = _flip(j)

                            # psrc/pdst point to source/destination
                            psrc = 0
                            pdst = 0
                            pend = pme1 - 1

                            while psrc <= pend:
                                # search for next FLIP'd entry
                                j = _flip(Iw[psrc])
                                psrc += 1
                                if j >= 0:
                                    Iw[pdst] = Pe[j]
                                    Pe[j] = pdst
                                    pdst += 1
                                    lenj = Len[j]
                                    # copy from source to destination
                                    knt3 = 0
                                    while knt3 <= lenj - 2:
                                        Iw[pdst] = Iw[psrc]
                                        pdst += 1
                                        psrc += 1
                                        knt3 += 1

                            # move the new partially-constructed element
                            p1 = pdst
                            psrc = pme1
                            while psrc <= pfree - 1:
                                Iw[pdst] = Iw[psrc]
                                pdst += 1
                                psrc += 1
                            pme1 = p1
                            pfree = pdst
                            pj = Pe[e]
                            p = Pe[me]

                        # ------------------------------------------------- #
                        # i is a principal variable not yet placed in Lme
                        # store i in new list
                        # ------------------------------------------------- #

                        # flag i as being in Lme by negating Nv [i]
                        degme += nvi
                        Nv[i] = -nvi
                        Iw[pfree] = i
                        pfree += 1

                        # remove variable i from degree link list
                        ilast = Last[i]
                        inext = Next[i]
                        if inext != EMPTY:
                            Last[inext] = ilast
                        if ilast != EMPTY:
                            Next[ilast] = inext
                        else:
                            # i is at the head of the degree list
                            Head[Degree[i]] = inext
                    knt2 += 1

                if e != me:
                    # set tree pointer and flag to indicate element e is
                    # absorbed into new element me (the parent of e is me)
                    Pe[e] = _flip(me)
                    W[e] = 0
                knt1 += 1

            pme2 = pfree - 1

        # ----------------------------------------------------------------- #
        # me has now been converted into an element in Iw [pme1..pme2]
        # ----------------------------------------------------------------- #

        # degme holds the external degree of new element
        Degree[me] = degme
        Pe[me] = pme1
        Len[me] = pme2 - pme1 + 1

        Elen[me] = _flip(nvpiv + degme)
        # FLIP (Elen (me)) is now the degree of pivot (including
        # diagonal part).

        # make sure that wflg is not too large.
        wflg = _clear_flag(wflg, wbig, W, n)

        # ================================================================= #
        # COMPUTE (W [e] - wflg) = |Le\Lme| FOR ALL ELEMENTS
        # ================================================================= #

        pme = pme1
        while pme <= pme2:
            i = Iw[pme]
            eln = Elen[i]
            if eln > 0:
                # note that Nv [i] has been negated to denote i in Lme:
                nvi = -Nv[i]
                wnvi = wflg - nvi
                p = Pe[i]
                while p <= Pe[i] + eln - 1:
                    e = Iw[p]
                    we = W[e]
                    if we >= wflg:
                        # unabsorbed element e has been seen in this loop
                        we -= nvi
                    elif we != 0:
                        # e is an unabsorbed element
                        # this is the first we have seen e in all of Scan 1
                        we = Degree[e] + wnvi
                    W[e] = we
                    p += 1
            pme += 1

        # ================================================================= #
        # DEGREE UPDATE AND ELEMENT ABSORPTION
        # ================================================================= #

        pme = pme1
        while pme <= pme2:
            i = Iw[pme]
            p1 = Pe[i]
            p2 = p1 + Elen[i] - 1
            pn = p1
            hash_ = 0       # unsigned Int
            deg = 0

            # scan the element list associated with supervariable i

            # UMFPACK/MA38-style approximate degree:
            if aggressive:
                p = p1
                while p <= p2:
                    e = Iw[p]
                    we = W[e]
                    if we != 0:
                        # e is an unabsorbed element
                        # dext = | Le \ Lme |
                        dext = we - wflg
                        if dext > 0:
                            deg += dext
                            Iw[pn] = e
                            pn += 1
                            hash_ = (hash_ + e) & _UINT_MASK
                        else:
                            # external degree of e is zero, absorb e into me
                            Pe[e] = _flip(me)
                            W[e] = 0
                    p += 1
            else:
                p = p1
                while p <= p2:
                    e = Iw[p]
                    we = W[e]
                    if we != 0:
                        # e is an unabsorbed element
                        dext = we - wflg
                        deg += dext
                        Iw[pn] = e
                        pn += 1
                        hash_ = (hash_ + e) & _UINT_MASK
                    p += 1

            # count the number of elements in i (including me):
            Elen[i] = pn - p1 + 1

            # scan the supervariables in the list associated with i
            p3 = pn
            p4 = p1 + Len[i]
            p = p2 + 1
            while p < p4:
                j = Iw[p]
                nvj = Nv[j]
                if nvj > 0:
                    # j is unabsorbed, and not in Lme.
                    # add to degree and add to new list
                    deg += nvj
                    Iw[pn] = j
                    pn += 1
                    hash_ = (hash_ + j) & _UINT_MASK
                p += 1

            # update the degree and check for mass elimination

            # with aggressive absorption, deg==0 is identical to the
            # Elen [i] == 1 && p3 == pn test, below.
            if Elen[i] == 1 and p3 == pn:
                # mass elimination
                Pe[i] = _flip(me)
                nvi = -Nv[i]
                degme -= nvi
                nvpiv += nvi
                nel += nvi
                Nv[i] = 0
                Elen[i] = EMPTY
            else:
                # update the upper-bound degree of i

                # the following degree does not yet include the size
                # of the current element, which is added later:
                Degree[i] = min(Degree[i], deg)

                # add me to the list for i

                # move first supervariable to end of list
                Iw[pn] = Iw[p3]
                # move first element to end of element part of list
                Iw[p3] = Iw[p1]
                # add new element, me, to front of list.
                Iw[p1] = me
                # store the new length of the list in Len [i]
                Len[i] = pn - p1 + 1

                # place in hash bucket.  Save hash key of i in Last [i].
                hash_ = hash_ % n       # unsigned modulo

                # if the Hhead array is not used:
                j = Head[hash_]
                if j <= EMPTY:
                    # degree list is empty, hash head is FLIP (j)
                    Next[i] = _flip(j)
                    Head[hash_] = _flip(i)
                else:
                    # degree list is not empty, use Last [Head [hash]] as
                    # hash head.
                    Next[i] = Last[j]
                    Last[j] = i

                Last[i] = hash_
            pme += 1

        Degree[me] = degme

        # Clear the counter array, W [...], by incrementing wflg.

        # make sure that wflg+n does not cause integer overflow
        lemax = max(lemax, degme)
        wflg += lemax
        wflg = _clear_flag(wflg, wbig, W, n)
        # at this point, W [0..n-1] < wflg holds

        # ================================================================= #
        # SUPERVARIABLE DETECTION
        # ================================================================= #

        pme = pme1
        while pme <= pme2:
            i = Iw[pme]
            if Nv[i] < 0:
                # i is a principal variable in Lme

                # examine all hash buckets with 2 or more variables.

                # let i = head of hash bucket, and empty the hash bucket
                hash_ = Last[i] & _UINT_MASK    # Int -> unsigned Int

                # if Hhead array is not used:
                j = Head[hash_]
                if j == EMPTY:
                    # hash bucket and degree list are both empty
                    i = EMPTY
                elif j < EMPTY:
                    # degree list is empty
                    i = _flip(j)
                    Head[hash_] = EMPTY
                else:
                    # degree list is not empty, restore Last [j] of head j
                    i = Last[j]
                    Last[j] = EMPTY

                while i != EMPTY and Next[i] != EMPTY:
                    # this bucket has one or more variables following i.
                    # scan all of them to see if i can absorb any entries
                    # that follow i in hash bucket.  Scatter i into w.
                    ln = Len[i]
                    eln = Elen[i]
                    # do not flag the first element in the list (me)
                    p = Pe[i] + 1
                    while p <= Pe[i] + ln - 1:
                        W[Iw[p]] = wflg
                        p += 1

                    # scan every other entry j following i in bucket
                    jlast = i
                    j = Next[i]

                    while j != EMPTY:
                        # check if j and i have identical nonzero pattern

                        # check if i and j have the same Len and Elen
                        ok = (Len[j] == ln) and (Elen[j] == eln)
                        # skip the first element in the list (me)
                        p = Pe[j] + 1
                        while ok and p <= Pe[j] + ln - 1:
                            if W[Iw[p]] != wflg:
                                ok = False
                            p += 1
                        if ok:
                            # found it!  j can be absorbed into i
                            Pe[j] = _flip(i)
                            # both Nv [i] and Nv [j] are negated since they
                            # are in Lme, and the absolute values of each
                            # are the number of variables in i and j:
                            Nv[i] += Nv[j]
                            Nv[j] = 0
                            Elen[j] = EMPTY
                            # delete j from hash bucket
                            j = Next[j]
                            Next[jlast] = j
                        else:
                            # j cannot be absorbed into i
                            jlast = j
                            j = Next[j]

                    # no more variables can be absorbed into i
                    # go to next i in bucket and clear flag array
                    wflg += 1
                    i = Next[i]
            pme += 1

        # ================================================================= #
        # RESTORE DEGREE LISTS AND REMOVE NONPRINCIPAL SUPERVARIABLES FROM
        # ELEMENT
        # ================================================================= #

        p = pme1
        nleft = n - nel
        pme = pme1
        while pme <= pme2:
            i = Iw[pme]
            nvi = -Nv[i]
            if nvi > 0:
                # i is a principal variable in Lme
                # restore Nv [i] to signify that i is principal
                Nv[i] = nvi

                # compute the external degree (add size of current element)
                deg = Degree[i] + degme - nvi
                deg = min(deg, nleft - nvi)

                # place the supervariable at the head of the degree list
                inext = Head[deg]
                if inext != EMPTY:
                    Last[inext] = i
                Next[i] = inext
                Last[i] = EMPTY
                Head[deg] = i

                # save the new degree, and find the minimum degree
                mindeg = min(mindeg, deg)
                Degree[i] = deg

                # place the supervariable in the element pattern
                Iw[p] = i
                p += 1
            pme += 1

        # ================================================================= #
        # FINALIZE THE NEW ELEMENT
        # ================================================================= #

        Nv[me] = nvpiv
        # save the length of the list for the new element me
        Len[me] = p - pme1
        if Len[me] == 0:
            # there is nothing left of the current pivot element
            # it is a root of the assembly tree
            Pe[me] = EMPTY
            W[me] = 0
        if elenme != 0:
            # element was not constructed in place: deallocate part of
            # it since newly nonprincipal variables may have been removed
            pfree = p

        # (Info statistics: write-only doubles, skipped)

    # ===================================================================== #
    # DONE SELECTING PIVOTS
    # ===================================================================== #

    # ===================================================================== #
    # POST-ORDERING
    # ===================================================================== #

    # restore Pe
    for i in range(n):
        Pe[i] = _flip(Pe[i])

    # restore Elen, for output information, and for postordering
    for i in range(n):
        Elen[i] = _flip(Elen[i])

    # compress the paths of the variables
    for i in range(n):
        if Nv[i] == 0:
            # i is an un-ordered row.  Traverse the tree from i until
            # reaching an element, e.
            j = Pe[i]
            if j == EMPTY:
                # Skip a dense variable.  It has no parent.
                continue

            # while (j is a variable)
            while Nv[j] == 0:
                j = Pe[j]
            # got to an element e
            e = j

            # traverse the path again from i to e, and compress the path
            j = i
            # while (j is a variable)
            while Nv[j] == 0:
                jnext = Pe[j]
                Pe[j] = e
                j = jnext

    # postorder the assembly tree
    amd_postorder(n, Pe, Nv, Elen,
                  W,                    # output order
                  Head, Next, Last)     # workspace

    # compute output permutation and inverse permutation

    # W [e] = k means that element e is the kth element in the new
    # order.  Use Head for inverse order.
    for k in range(n):
        Head[k] = EMPTY
        Next[k] = EMPTY
    for e in range(n):
        k = W[e]
        if k != EMPTY:
            Head[k] = e

    # construct output inverse permutation in Next,
    # and permutation in Last
    nel = 0
    for k in range(n):
        e = Head[k]
        if e == EMPTY:
            break
        Next[e] = nel
        nel += Nv[e]

    # order non-principal variables (dense, & those merged into supervar's)
    for i in range(n):
        if Nv[i] == 0:
            e = Pe[i]
            if e != EMPTY:
                # This is an unordered variable that was merged
                # into element e via supernode detection or mass
                # elimination of i when e became the pivot element.
                # Place i in order just before e.
                Next[i] = Next[e]
                Next[e] += 1
            else:
                # This is a dense unordered variable, with no parent.
                # Place it last in the output order.
                Next[i] = nel
                nel += 1

    for i in range(n):
        k = Next[i]
        Last[k] = i

    if stats is not None:
        # integer counterparts of Info [AMD_NDENSE] and Info [AMD_NCMPA]
        stats["ndense"] = ndense
        stats["ncmpa"] = ncmpa


# ========================================================================= #
# === AMD_postorder ======================================================= #
# ========================================================================= #

def amd_postorder(nn, Parent, Nv, Fsize, Order, Child, Sibling, Stack):
    for j in range(nn):
        Child[j] = EMPTY
        Sibling[j] = EMPTY

    # place the children in link lists - bigger elements tend to be last
    j = nn - 1
    while j >= 0:
        if Nv[j] > 0:
            # this is an element
            parent = Parent[j]
            if parent != EMPTY:
                # place the element in link list of the children its parent
                # bigger elements will tend to be at the end of the list
                Sibling[j] = Child[parent]
                Child[parent] = j
        j -= 1

    # place the largest child last in the list of children for each node
    for i in range(nn):
        if Nv[i] > 0 and Child[i] != EMPTY:
            # find the biggest element in the child list
            fprev = EMPTY
            maxfrsize = EMPTY
            bigfprev = EMPTY
            bigf = EMPTY
            f = Child[i]
            while f != EMPTY:
                frsize = Fsize[f]
                if frsize >= maxfrsize:
                    # this is the biggest seen so far
                    maxfrsize = frsize
                    bigfprev = fprev
                    bigf = f
                fprev = f
                f = Sibling[f]

            fnext = Sibling[bigf]

            if fnext != EMPTY:
                # if fnext is EMPTY then bigf is already at the end of list
                if bigfprev == EMPTY:
                    # delete bigf from the element of the list
                    Child[i] = fnext
                else:
                    # delete bigf from the middle of the list
                    Sibling[bigfprev] = fnext

                # put bigf at the end of the list
                Sibling[bigf] = EMPTY
                Sibling[fprev] = bigf

    # postorder the assembly tree
    for i in range(nn):
        Order[i] = EMPTY

    k = 0

    for i in range(nn):
        if Parent[i] == EMPTY and Nv[i] > 0:
            k = amd_post_tree(i, k, Child, Sibling, Order, Stack)


# ========================================================================= #
# === AMD_post_tree ======================================================= #
# ========================================================================= #

def amd_post_tree(root, k, Child, Sibling, Order, Stack):
    # non-recursive version, using an explicit stack

    # push root on the stack
    head = 0
    Stack[0] = root

    while head >= 0:
        # get head of stack
        i = Stack[head]

        if Child[i] != EMPTY:
            # the children of i are not yet ordered
            # push each child onto the stack in reverse order
            # so that small ones at the head of the list get popped first
            # and the biggest one at the end of the list gets popped last
            f = Child[i]
            while f != EMPTY:
                head += 1
                f = Sibling[f]
            h = head
            f = Child[i]
            while f != EMPTY:
                Stack[h] = f
                h -= 1
                f = Sibling[f]

            # delete child list so that i gets ordered next time we see it
            Child[i] = EMPTY
        else:
            # the children of i (if there were any) are already ordered
            # remove i from the stack and order it.  Front i is kth front
            head -= 1
            Order[i] = k
            k += 1
    return k


# ========================================================================= #
# === AMD_order =========================================================== #
# ========================================================================= #

def amd_order(n, Ap, Ai, stats=None):
    """Port of ``amd_l_order(n, Ap, Ai, P, NULL, Info)``.

    Returns ``(status, perm)`` where ``perm`` is what C writes to ``P``
    (a list of length n).  For an error status (``AMD_INVALID`` or
    ``AMD_OUT_OF_MEMORY``) C leaves ``P`` untouched and ``perm`` is ``None``.
    For ``n == 0`` C returns ``AMD_OK`` without touching ``P``; ``perm`` is
    then the empty list.

    ``stats`` (optional dict) is a diagnostic-only output receiving the
    integer counters ``ndense`` (dense rows removed), ``ncmpa`` (garbage
    collections), ``nzaat`` and ``jumbled``; it never affects the ordering.
    """
    # make sure inputs exist and n is >= 0
    if Ai is None or Ap is None or n < 0:
        return AMD_INVALID, None    # arguments are invalid

    if n == 0:
        return AMD_OK, []           # n is 0 so there's nothing to do

    Ap = [int(v) for v in Ap]
    Ai = [int(v) for v in Ai]

    nz = Ap[n]
    if nz < 0:
        return AMD_INVALID, None

    # check if n or nz will cause size_t overflow
    if (n >= _SIZE_T_MAX // _SIZEOF_INT
            or nz >= _SIZE_T_MAX // _SIZEOF_INT):
        return AMD_OUT_OF_MEMORY, None      # problem too large

    # check the input matrix: AMD_OK, AMD_INVALID, or AMD_OK_BUT_JUMBLED
    status = amd_valid(n, n, Ap, Ai)

    if status == AMD_INVALID:
        return AMD_INVALID, None    # matrix is invalid

    # allocate two size-n integer workspaces
    Len = [0] * n
    Pinv = [0] * n
    P = [0] * n

    if status == AMD_OK_BUT_JUMBLED:
        # sort the input matrix and remove duplicate entries
        Rp = [0] * (n + 1)
        Ri = [0] * nz
        # use Len and Pinv as workspace to create R = A'
        amd_preprocess(n, Ap, Ai, Rp, Ri, Len, Pinv)
        Cp = Rp
        Ci = Ri
    else:
        # order the input matrix as-is.  No need to compute R = A' first
        Cp = Ap
        Ci = Ai

    # determine the symmetry and count off-diagonal nonzeros in A+A'
    nzaat = amd_aat(n, Cp, Ci, Len, P)

    # allocate workspace for matrix, elbow room, and 6 size-n vectors
    slen = nzaat                                    # space for matrix
    ok = ((slen + nzaat // 5) & _SIZE_T_MAX) >= slen   # size_t overflow
    slen = (slen + nzaat // 5) & _SIZE_T_MAX        # add elbow room
    i = 0
    while ok and i < 7:
        ok = ((slen + n) & _SIZE_T_MAX) > slen      # size_t overflow
        slen = (slen + n) & _SIZE_T_MAX     # size-n elbow room, 6 size-n work
        i += 1
    ok = ok and (slen < _SIZE_T_MAX // _SIZEOF_INT)     # check for overflow
    ok = ok and (slen < AMD_INT_MAX)        # S[i] for Int i must be OK
    if not ok:
        # :: out of memory :: (or problem too large)
        return AMD_OUT_OF_MEMORY, None

    # order the matrix
    if stats is not None:
        stats["nzaat"] = nzaat
        stats["jumbled"] = int(status == AMD_OK_BUT_JUMBLED)
    amd_1(n, Cp, Ci, P, Pinv, Len, slen, stats)

    return status, P    # successful ordering
