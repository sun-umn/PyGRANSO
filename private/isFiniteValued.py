import torch
import numpy as np

def isFiniteValued(A):
    """ 
    isFiniteValued:
      Checks whether A only has finite values (no nans/infs).
    """
    if isinstance(A, float) and A < np.inf:
      return True

    tf =  torch.isfinite(torch.norm(A,1)).item()

    return tf