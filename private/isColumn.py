def isColumn(c):
    """  
    isColumn:
        Checks whether c is a nonempty column vector.
    """
    [m,n]   = c.shape
    tf      = m > 0 and n == 1

    return tf