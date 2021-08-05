from private.isARealNumber import isARealNumber
import math

def isAnInteger(x):
    """
    isAnInteger:
       Returns whether or not x is an integer.  x must be finite.
    """
    tf =    isARealNumber(x) and math.isfinite(x) and  round(x) == x

    return tf