import math
import scipy.linalg as la
import numpy as np
from numpy import conjugate as conj

def regularizePosDefMatrix(  A, condnum_limit, limit_max_eigenvalues ):
    """
    regularizePosDefMatrix:
        Regularizes a dense positive definite matrix A so that its 
        condition number never exceeds threshold.  If regularization is 
        needed, i.e. cond(A) > threshold, its eigenvalue decomposition is 
        formed and the regularization is done by either shifting the 
        largest eigenvalue(s) downward or by shifting the smallest 
        eigenvalue(s) upward, so that the new eigenvalues satisfy 
        lambda_max / lambda_min == threshold.

        INPUT:
            A           
                dense positive definite matrix
                - if A is not positive definite, result will be wrong
                - if A is sparse, output will be unchanged sparse A

            threshold   
                maximum condition number of A allowed  
                - regularization only happens for cond(A) > threshold
                - may be set to inf (no regularization ever)

            limit_max_eigenvalues
                - true: shift largest eigenvalues downward 
                - false: shift smalleset eigenvalues upward

        OUTPUT:
            Areg  
                possibly regularized version of A
                cond(Areg) <= threshold always holds.

            info
                Integer code in  {0,1,2} indicating the result
                0       A was regularized
                1       A was not regularized, either because its condition
                        number did not exceed the threshold condnum_limit or 
                        because A was a sparse matrix.  
                2       eig threw an error       

        If you publish work that uses or refers to PyGRANSO, please cite both
        PyGRANSO and GRANSO paper:

        [1] Buyun Liang, Tim Mitchell, and Ju Sun,
            NCVX: A User-Friendly and Scalable Package for Nonconvex
            Optimization in Machine Learning, arXiv preprint arXiv:2111.13984 (2021).
            Available at https://arxiv.org/abs/2111.13984

        [2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton,
            A BFGS-SQP method for nonsmooth, nonconvex, constrained
            optimization and its evaluation using relative minimization
            profiles, Optimization Methods and Software, 32(1):148-181, 2017.
            Available at https://dx.doi.org/10.1080/10556788.2016.1208749
            
        regularizePosDefMatrix.py (introduced in PyGRANSO v1.0.0)
        Copyright (C) 2016-2021 Tim Mitchell

        This file is a direct port of regularizePosDefMatrix.m, which is included as part
        of GRANSO v1.6.4 and from URTM (http://www.timmitchell/software/URTM).
        Ported from MATLAB to Python by Buyun Liang 2021

        For comments/bug reports, please visit the PyGRANSO webpage:
        https://github.com/sun-umn/PyGRANSO

        =========================================================================
        |  PyGRANSO: A PyTorch-enabled port of GRANSO with auto-differentiation |
        |  Copyright (C) 2021 Tim Mitchell and Buyun Liang                      |
        |                                                                       |
        |  This file is part of PyGRANSO.                                       |
        |                                                                       |
        |  PyGRANSO is free software: you can redistribute it and/or modify     |
        |  it under the terms of the GNU Affero General Public License as       |
        |  published by the Free Software Foundation, either version 3 of       |
        |  the License, or (at your option) any later version.                  |
        |                                                                       |
        |  PyGRANSO is distributed in the hope that it will be useful,          |
        |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
        |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
        |  GNU Affero General Public License for more details.                  |
        |                                                                       |
        |  You should have received a copy of the GNU Affero General Public     |
        |  License along with this program.  If not, see                        |
        |  <http://www.gnu.org/licenses/agpl.html>.                             |
        =========================================================================
    """
    #  regularizes dense positive definite matrices
    Areg = A  
    info = 1 # didn't regularize

    print("TODO: implement issparse(Areg)")
    # if issparse(Areg) || isinf(condnum_limit)
    if math.isinf(condnum_limit):
        return [Areg,info]
    
   
    try: 
        # Matlab: [V, D] = eig(A)
        [D,V] = la.eig(A)
        d = np.diag(D)
    except:
        info = 2
        return [Areg,info]
    
    if limit_max_eigenvalues:
        [d,updated] = lowerLargestEigenvalues(d,condnum_limit)
    else:
        [d,updated] = raiseSmallestEigenvalues(d,condnum_limit)
        
    if updated:
        Areg = V@np.diag(d) @ conj(V.T)
        info = 0

    return [Areg,info]

    
def raiseSmallestEigenvalues(d,condnum_limit):
    #  Even though A should be positive definite theoretically (BFGS), if 
    #  min(d) is tiny, it may be that min(d) is, numerically, negative 
    #  (or zero).  However, the following works in that case too and should
    #  also numerically restore positive definiteness, that is, the new set 
    #  of eigenvalues will all be strictly positive.

    max_eval        = np.max(d)  # assume this is positive
    new_min_eval    = max_eval / condnum_limit
    indx            = d < new_min_eval
    d[indx]         = new_min_eval
    updated         = np.any(indx != False)
    return [d,updated]

def lowerLargestEigenvalues(d,condnum_limit):

    #  get smallest modulus of eigenvalues
    min_mod_of_evals    = min(abs(d))

    if min_mod_of_evals > 0: 
        #  lower the largest eigenvalues to make the regularized version of 
        #  A have a condition number that is equal to cond_number_limit
        new_max_eval    = min_mod_of_evals * condnum_limit
        indx            = d > new_max_eval
        d[indx]         = new_max_eval
        updated         = np.any(indx != False)
    else:
        #  at least one eigenvalue is exactly zero so A can't be regularized
        #  by lowering the largest eigenvalues.  Instead, in this extremely 
        #  unlikely event, we will resort to regularizing A by increasing 
        #  the smallest eigenvalues so that they are all strictly positive.
        [d,updated]     = raiseSmallestEigenvalues(d,condnum_limit)
    
    return [d,updated]