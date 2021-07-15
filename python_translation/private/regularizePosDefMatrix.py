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