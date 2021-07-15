def truncate(s,width):
    """
   truncate:
       Truncate string so that it's length is at most width chars long.
    """ 
    s = s[0:min(len(s),width)]
    return s