# import torch
from numpy import linalg as LA

def isPositiveDefinite(A):
    """  
    isPositiveDefinite:
      Checks whether A is a positive definite by attempting to form its
      Cholesky factorization.  Returns true if chol() succeeds, false 
      otherwise.
    """
    try :
        # torch.linalg.cholesky(A)
        A = A.cpu()
        LA.cholesky(A)
    except: 
        tf = False
        return tf
    
    tf = True

    return tf