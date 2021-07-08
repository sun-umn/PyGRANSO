import math
from private import nDigitsInWholePart as nDIWP

def double2FixedWidthStr(width, str_in = None):
    """
    double2FixedWidthStr:
      Formats number x in the desired width field so that the length of
      the string remains constant for all values of x and the most
      significant digit is always in the second place (so sign changes do
      not shift alignment and a '+' is not necessary).  The precision
      displayed is always at least 2 digits and varies on the magnitude
      of the number, as numbers requiring scientific notation require
      more characters to display their magnitude.
    """
    #  need places for
    #  - sign (1 char)
    #  - minimum of 2 digits with decimal point (3 chars)
    #  - up to 5 chars for exponent e+111
    MIN_WIDTH   = 9

    #  must be able to display up to 16 digits
    #  - sign (1 char)
    #  - 16 digits + decimal point (17 chars)
    #  - up to 5 chars for exponent
    MAX_WIDTH   = 23
    
    def assertWidth(width):
        assert width >= MIN_WIDTH and width <= MAX_WIDTH, "double2FixedWidthStr invalidInput : double2FixedWidthStr: width must be in [9,10,...,23]."


    if str_in == None:
        assertWidth(width)
        out     = lambda x: double2FixedWidth(x,width)
    else:
        assertWidth(width)
        out     = double2FixedWidth(str_in,width)

    return out
    
    


def double2FixedWidth(x,width):
    if not str.isnumeric(x):
        x_str       = " n/a%s"%(" "*(width-4))
        return
    elif x == 0:
        x_str       = " 0.%s"%("0"*(width-3))
        return
    elif x == float("inf"):
        x_str       = " Inf%s"%(" "*(width-4))  
        return
    elif x == -float("inf"):
        x_str       = "-Inf%s"%(" "*(width-4)) 
        return
    elif math.isnan(x):
        x_str       = " NaN%s"%(" "*(width-4)) 
        return

    is_negative     = x < 0
    x               = abs(x)
    n_whole         = nDIWP.nDigitsInWholePart(x)

    if x >= 1e+100 or x < 1e-99:
        #  this assumes x is positive
        x_str       = "%.*e"%(width - 8,x) 
    elif n_whole > width - 3 or x < 1e-3:
        #  this assumes x is positive
        x_str       = "%.*e"%(width - 7,x)
    else:
        digits      = width - 2 - n_whole
        if n_whole < 1:
            digits  = digits - 1
        
    
        x_str       = "%.*f"%(digits,x)  
        #  get length without sign since the actual number of whole digits 
        #  may be increased by one due to rounding in sprintf (e.g.
        #  99.99999999...)
        expected    = width - float(x >= 0)  
        if len(x_str) > expected:
            x_str   = x_str[0:expected]
        
    
    if is_negative:
        x_str       = "-" + x_str  
    else:
        x_str       = " " + x_str 

    return x_str