import math

def nDigitsInWholePart(number):
    """
   nDigitsInWholePart:
       Returns the number of the digits in the whole part of a number.
       This function returns a positive integer since 0 has the minimum
       number of digits, namely one.
    """
    d = max(math.floor(math.log10(abs(number))) + 1, 1)
    return d