from dbg_print import dbg_print
import numpy as np
import torch

def isRealValued(X):
    """  
    isRealValued:
      Checks whether X is real-valued, that is, it either has no
      imaginary part or it's imaginary part is exactly zero.  

      Note that MATLAB's isreal function only checks whether there is an
      imaginary part allocated, i.e., isreal returns false if an
      imaginary part is allocated, even if it is zero.
    """
    dbg_print("private.isRealValued: Skip imag validate")
    # tf = np.isreal(X) or not np.any(np.imag(X))

    if isinstance(X, float) or isinstance(X, int):
      tf = np.isreal(X)
    else:
      tf = torch.all(torch.isreal(X) == True)  

    return tf