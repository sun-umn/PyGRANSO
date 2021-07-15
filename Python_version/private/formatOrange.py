def formatOrange(s):
    """
    formatOrange:
      formatOrange applies undocumented formatting symbols to a string
      such that fprintf will print string s in orange text.
    """
    O  = '\033[33m' # orange
    W  = '\033[0m'  # white (normal)
    s = (O + s + W)
    return s