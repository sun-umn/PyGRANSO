import numpy.linalg as LA
import numpy as np

def isFiniteValued(A):
    """ 
    isFiniteValued:
      Checks whether A only has finite values (no nans/infs).
    """

    #  Using norm(A,1) is much quicker than using any of:
    #    any(), isinf(), isnan(), A(:)
    tf = np.isnumeric(A) and np.isfinite(LA.norm(A,1))

    return tf