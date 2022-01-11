import numpy as np

def wrapToLines(str_in,width,indent):
    """
    wrapToLines:
        Converts the string str_in into a cell array of lines, each of at most
        length width by doing basic word wrapping (no hyphenation).  Each
        line can be optionally indented.  Words will be truncated if their 
        length exceeds the specified width minus the indent value.

        If you publish work that uses or refers to PyGRANSO, please cite both
        PyGRANSO and GRANSO paper:

        [1] Buyun Liang, Tim Mitchell, and Ju Sun,
            NCVX: A User-Friendly and Scalable Package for Nonconvex
            Optimization in Machine Learning, arXiv preprint arXiv:2111.13984 (2021).
            Available at https://arxiv.org/abs/2111.13984

        [2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton,
            A BFGS-SQP method for nonsmooth, nonconvex, constrained
            optimization and its evaluation using relative minimization
            profiles, Optimization Methods and Software, 32(1):148-181, 2017.
            Available at https://dx.doi.org/10.1080/10556788.2016.1208749
            
        wrapToLines.py (introduced in PyGRANSO v1.0.0)
        Copyright (C) 2016-2021 Tim Mitchell

        This file is a direct port of wrapToLines.m, which is included as part
        of GRANSO v1.6.4 and from URTM (http://www.timmitchell/software/URTM).
        Ported from MATLAB to Python by Buyun Liang 2021

        For comments/bug reports, please visit the PyGRANSO webpage:
        https://github.com/sun-umn/PyGRANSO

        =========================================================================
        |  PyGRANSO: A PyTorch-enabled port of GRANSO with auto-differentiation |
        |  Copyright (C) 2021 Tim Mitchell and Buyun Liang                      |
        |                                                                       |
        |  This file is part of PyGRANSO.                                       |
        |                                                                       |
        |  PyGRANSO is free software: you can redistribute it and/or modify     |
        |  it under the terms of the GNU Affero General Public License as       |
        |  published by the Free Software Foundation, either version 3 of       |
        |  the License, or (at your option) any later version.                  |
        |                                                                       |
        |  PyGRANSO is distributed in the hope that it will be useful,          |
        |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
        |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
        |  GNU Affero General Public License for more details.                  |
        |                                                                       |
        |  You should have received a copy of the GNU Affero General Public     |
        |  License along with this program.  If not, see                        |
        |  <http://www.gnu.org/licenses/agpl.html>.                             |
        =========================================================================
    """
    if not isinstance(str_in,str):
        print(  "wrapToLines invalidInput: wrapToLines: str_in must be a string of characters."  )
    
    if width < 4:
        print(  "wrapToLines invalidInput: wrapToLines: width must be at least 4."  )
        
    if indent < 0:
        print(  "wrapToLines invalidInput: wrapToLines:  indent must be nonnegative."  )
    
    if width - indent < 4: 
        print(  "wrapToLines invalidInput: wrapToLines:  width - indent must be at least 4."  )
    
    chars       = len(str_in)
    max_lines   = np.ceil(chars/width)

    words       = strToTrimmedWords(str_in,width)
    
    count       = 0
    lines       = [""] * int(max_lines)
    line        = getBlankLine(width)
    position    = 1 + indent
    for j in range(len(words)):
        word        = words[j]
        chars       = len(word) 
        word_end    = position - 1 + chars
        if word_end > width:
            count           = count + 1
            lines[count-1]    = line
            line            = getBlankLine(width)
            position        = 1 + indent
            word_end        = position - 1 + chars
        
        line_lst = list(line)
        word_lst = list(word)
        ii = 0
        for i in range(position-1,word_end):
            line_lst[i] = word_lst[ii]
            ii += 1
        line = "".join(line_lst)
      
        if j == len(words)-1:
            count           = count + 1
            lines[count-1]    = line
        elif word[-1] == "." or word[-1] == "?" or word[-1] == "!":
            position = position + chars + 2
        else:
            position = position + chars + 1
        
    lines = lines[0:count]
    return lines


def getBlankLine(width):
    line = " " * width
    return line

def trimWord(w,width):
    if len(w) < 5:
        return w
    
    w_lst = list(w)
    w_lst               = w[0:width]
    for i in range(-1,-4):
        w_lst = "."
    w = "".join(w_lst)
    return w

def strToTrimmedWords(str_in,width):
    words = str_in.split()

    return words