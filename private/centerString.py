import math

def centerString(s,width):
    """
   centerString:
       Centers string s in the specified width via adding spaces to each 
       side.  If s is longer than width, s will be truncated. If an odd
       number of spaces is needed, the left side will have one more space
       compared to the right side.
    """

    s                   = s.strip()
    available_spaces    = width - len(s)
    
    if available_spaces > 0: 
        spaces          = math.floor(available_spaces / 2)
        spaces_str      = " " * spaces
        if available_spaces % 2 == 0:
            s           = spaces_str + s + spaces_str
        else:
            s           = " " + spaces_str + s + spaces_str
    else:
        s               = s[0:width]

    return s