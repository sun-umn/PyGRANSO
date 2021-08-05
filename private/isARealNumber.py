import numpy as np
from private.isRealValued import isRealValued
from dbg_print import dbg_print

def isARealNumber(x):
    """
    isARealNumber:
      Returns whether or not x is a real number.  
      +Inf or -Inf are considered real numbers.  
      NaNs are not considered real numbers.
    """
    # tf =    np.isscalar(x)  and np.isnumeric(x) and not np.isnan(x) and isRealValued(x)
    tf =    np.isscalar(x)  and not np.isnan(x) and isRealValued(x)
    dbg_print("private.isRealNumber : Skip isnumeric for now")

    return tf