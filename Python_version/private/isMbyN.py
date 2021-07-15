def isMbyN(A,m,n):
    """  
    isMbyN:
      Checks whether A is an M by N matrix.
    """
    tf = A.shape[0] == m and A.shape[1] == n 

    return tf