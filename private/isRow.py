def isRow(r):
    """
    isRow:
      Checks whether r is a nonempty row vector.
    """
    [m,n]   = r.shape
    tf      = m == 1 and n > 0

    return tf