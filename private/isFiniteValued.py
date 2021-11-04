import torch

def isFiniteValued(A):
    """ 
    isFiniteValued:
      Checks whether A only has finite values (no nans/infs).
    """

    tf =  torch.isfinite(torch.norm(A,1)).item()

    return tf