def isColumn(c):
    """  
    isColumn:
        Checks whether c is a nonempty column vector, that is, for 
        [m,n] = size(r), both m > 0 and n == 1 must hold.
    """
    [m,n]   = c.shape
    tf      = m > 0 and n == 1

    return tf