import numpy as np
from private.isRealValued import isRealValued

def isARealNumber(x):
    """
    isARealNumber:
      Returns whether or not x is a real number.  
      +Inf or -Inf are considered real numbers.  
      NaNs are not considered real numbers.
    """
    tf =    np.isscalar(x)  and not np.isnan(x) and isRealValued(x)

    return tf