from torch import linalg as LA
from dbg_print import dbg_print
import torch

def isFiniteValued(A):
    """ 
    isFiniteValued:
      Checks whether A only has finite values (no nans/infs).
    """

    #  Using norm(A,1) is much quicker than using any of:
    #    any(), isinf(), isnan(), A(:)
    # tf = np.isnumeric() and np.isfinite(LA.norm(A,1))

    tf =  torch.all(torch.isfinite(LA.norm(A,1)))
    dbg_print("private.isFiniteValued : Skip isnumeric for now")

    return tf