def isRow(r):
    """
    isRow:
      Checks whether r is a nonempty row vector, that is, for 
      [m,n] = size(r), both m = 1 and n > 0 must hold.
    """
    [m,n]   = r.shape
    tf      = m == 1 and n > 0

    return tf