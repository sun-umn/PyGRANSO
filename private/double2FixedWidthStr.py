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

        USAGE:
            1) Get a function handle to do formatting for a given width
            format_fn   = double2FixedWidth(width);
        
            2) Format a number to a string of a given width
            str         = double2FixedWidth(number,width);
        
        INPUT:
            width       An integer between 9 and 23 specifying the number
                        of chars to use for representing a number as a string.
                        For width == 9, the string representation will always
                        show the first two most significant digits while width
                        == 23, the representation will show all the significant
                        digits up to machine precision.
        
            number      A double to convert to a fixed width string.
        
        OUTPUT:         [Either one of the following] 
            format_fn   Function handle that takes single argument x and
                        converts it to a string representation with fixed width
                        such that the most significant digit will always be the
                        second character.  Note that if x is a number, variable
                        precision is used to ensure the width of the string
                        representation does not change.  Nonnumeric x will
                        cause a centered 'n/a' to be returned.  Both infs 
                        (positive and negative) and NaNs are supported.
        
            str         fixed width aligned string representation of number. 

        If you publish work that uses or refers to NCVX, please cite both
        NCVX and GRANSO paper:

        [1] Buyun Liang, and Ju Sun. 
            NCVX: A User-Friendly and Scalable Package for Nonconvex 
            Optimization in Machine Learning. arXiv preprint arXiv:2111.13984 (2021).
            Available at https://arxiv.org/abs/2111.13984

        [2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton 
            A BFGS-SQP method for nonsmooth, nonconvex, constrained 
            optimization and its evaluation using relative minimization 
            profiles, Optimization Methods and Software, 32(1):148-181, 2017.
            Available at https://dx.doi.org/10.1080/10556788.2016.1208749
            
        Change Log:
            double2FixedWidthStr.m introduced in GRANSO Version 1.0
            
            Buyun Dec 20, 2021 (NCVX Version 1.0.0):
                double2FixedWidthStr.py is translated from double2FixedWidthStr.m in GRANSO Version 1.6.4. 

        For comments/bug reports, please visit the NCVX webpage:
        https://github.com/sun-umn/NCVX
        
        NCVX Version 1.0.0, 2021, see AGPL license info below.

        =========================================================================
        |  double2FixedWidthStr.m                                               |
        |  Copyright (C) 2016 Tim Mitchell                                      |
        |                                                                       |
        |  This file is originally from URTM.                                   |
        |                                                                       |
        |  URTM is free software: you can redistribute it and/or modify         |
        |  it under the terms of the GNU Affero General Public License as       |
        |  published by the Free Software Foundation, either version 3 of       |
        |  the License, or (at your option) any later version.                  |
        |                                                                       |
        |  URTM is distributed in the hope that it will be useful,              |
        |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
        |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
        |  GNU Affero General Public License for more details.                  |
        |                                                                       |
        |  You should have received a copy of the GNU Affero General Public     |
        |  License along with this program.  If not, see                        |
        |  <http://www.gnu.org/licenses/agpl.html>.                             |
        =========================================================================

        =========================================================================
        |  GRANSO: GRadient-based Algorithm for Non-Smooth Optimization         |
        |  Copyright (C) 2016 Tim Mitchell                                      |
        |                                                                       |
        |  This file is translated from GRANSO.                                 |
        |                                                                       |
        |  GRANSO is free software: you can redistribute it and/or modify       |
        |  it under the terms of the GNU Affero General Public License as       |
        |  published by the Free Software Foundation, either version 3 of       |
        |  the License, or (at your option) any later version.                  |
        |                                                                       |
        |  GRANSO is distributed in the hope that it will be useful,            |
        |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
        |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
        |  GNU Affero General Public License for more details.                  |
        |                                                                       |
        |  You should have received a copy of the GNU Affero General Public     |
        |  License along with this program.  If not, see                        |
        |  <http://www.gnu.org/licenses/agpl.html>.                             |
        =========================================================================

        =========================================================================
        |  NCVX (NonConVeX): A User-Friendly and Scalable Package for           |
        |  Nonconvex Optimization in Machine Learning.                          |
        |                                                                       |
        |  Copyright (C) 2021 Buyun Liang                                       |
        |                                                                       |
        |  This file is part of NCVX.                                           |
        |                                                                       |
        |  NCVX is free software: you can redistribute it and/or modify         |
        |  it under the terms of the GNU Affero General Public License as       |
        |  published by the Free Software Foundation, either version 3 of       |
        |  the License, or (at your option) any later version.                  |
        |                                                                       |
        |  GRANSO is distributed in the hope that it will be useful,            |
        |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
        |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
        |  GNU Affero General Public License for more details.                  |
        |                                                                       |
        |  You should have received a copy of the GNU Affero General Public     |
        |  License along with this program.  If not, see                        |
        |  <http://www.gnu.org/licenses/agpl.html>.                             |
        =========================================================================
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
    if x == 0:
        x_str       = " 0.%s"%("0"*(width-3))
        return x_str
    elif x == float("inf"):
        x_str       = " Inf%s"%(" "*(width-4))  
        return x_str
    elif x == -float("inf"):
        x_str       = "-Inf%s"%(" "*(width-4)) 
        return x_str
    elif math.isnan(x):
        x_str       = " NaN%s"%(" "*(width-4)) 
        return x_str
    elif not isinstance(x,int)  and not isinstance(x,float) and x.dtype != "int64" and x.dtype != "float32":
        x_str       = " n/a%s"%(" "*(width-4))
        return x_str

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
        expected    = int(width - float(x >= 0)  )
        if len(x_str) > expected:
            x_str   = x_str[0:expected]
        
    if is_negative:
        x_str       = "-" + x_str  
    else:
        x_str       = " " + x_str 

    return x_str