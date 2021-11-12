def printOrange(strPrint):
    """
    printOrange makes the desired printing to be orange
    """
    O  = '\033[33m' # orange
    W  = '\033[0m'  # white (normal)
    print(O + strPrint + W, end="")