from private import printOrange as pO

def printMessageBox(use_ascii,use_orange,margin_spaces,title_top,title_bottom,msg_lines,sides=True,user_width=0):
    """
    printMessageBox:
        Prints a message to the console surrounded by a border.
        If user_width is not provided, the border is automatically sized to
        fit the maximum width line (including given whitespace formatting)
        specified in the message.

        INPUT:
            use_ascii       logical: use standard ascii instead of extended
                            characters to make the border
    
            use_orange      logical: use orange text to print the border and 
                            message
    
            margin_spaces   nonnegative integer: number of spaces for left and
                            right margins of the text
    
            title_top       string or []: specify a title for the top border. 
                            Empty or a blank string means no title.
    
            title_bottom    string or []: specify a title for the top border. 
                            Empty or a blank string means no title.
    
            msg_lines       cell array of each line to print in the message
                            box.  White space formatting is observed.
    
            sides           optional argument, logical: whether or not side 
                            borders should be printed.  
            
            user_width      optional argument, positive integer such that:
                                user_width must be >= 6 * 2*margin_spaces
                            to specify the width of the box.  Note that the box
                            width may increase beyond this value to accommodate
                            top and bottom titles, if they would otherwise not
                            completely fit.  The box's content however is
                            treated differently.  If sides == true, lines in
                            msg_lines that are too long will be truncated as
                            necessary.  If sides == false, user_width is only
                            used to determine the widths of the top/bottom
                            borders and lines will be not truncated, even if
                            they extend beyond the widths of the headers.

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
            printMessageBox.m introduced in GRANSO Version 1.0
            
            Buyun Dec 20, 2021 (NCVX Version 1.0.0):
                printMessageBox.py is translated from printMessageBox.m in GRANSO Version 1.6.4. 

        For comments/bug reports, please visit the NCVX webpage:
        https://github.com/sun-umn/NCVX
        
        NCVX Version 1.0.0, 2021, see AGPL license info below.

        =========================================================================
        |  printMessageBox.m                                                    |
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

    TITLE_INDENT        = 5
    TITLE_MARGIN        = 1
   
    have_user_width = (user_width != 0)
    border_spaces       = 2*margin_spaces
    fixed_width_headers = have_user_width

    #  process title and line widths
    line_width_arr = [len(msg_line) for msg_line in msg_lines]
    max_line_width      = max(line_width_arr)
    # strtrim in matlab -> strip in python
    title_top           = title_top.strip()
    title_bottom        = title_bottom.strip()
    title_top_width     = len(title_top)
    title_bottom_width  = len(title_bottom)
    
    # min width of header, not including the ends (the corner chars)
    header_width        = max(title_top_width,title_bottom_width)
    if header_width > 0:  
        header_width    = header_width + 2*(TITLE_INDENT + TITLE_MARGIN)
    
    header_width        = max(header_width,user_width)
    if not fixed_width_headers:
        header_width    = max(header_width,max_line_width+border_spaces)
    
    if not have_user_width:
        # increase max_line_width if the header width is longer
        max_line_width  = max(max_line_width,header_width-border_spaces)
    elif sides:
        # crop lines if they are too long for the user-specified box
        max_line_width  = header_width - border_spaces
        trunc_fn        = lambda s: truncateStr(s,max_line_width)
        msg_lines = [trunc_fn(msg_line) for msg_line in msg_lines]

    [top,bottom,vbar]   = getHeaders(use_ascii,header_width)
    # insert titles as necessary                                
    add_title_fn        = lambda h,t: addTitle(h,t,TITLE_INDENT,TITLE_MARGIN)
    top                 = add_title_fn(top,title_top)
    bottom              = add_title_fn(bottom,title_bottom)    
    
    if not sides:
        vbar            = ""
    
    format_str          = lineFormatStr(max_line_width,margin_spaces,vbar)
    
    if use_orange and not use_ascii:
        print_fn        = lambda strPrint : pO.printOrange(strPrint)
    else:
        print_fn        = lambda strPrint : print(strPrint,end="")
    
    print_line_fn       = lambda str_tmp: print_fn(format_str % (str_tmp) )
    
    print_fn(top)
    for msg_line in msg_lines:
        print_line_fn(msg_line)
    
    print_fn(bottom)

def truncateStr(s,width):
    if len(s) > width:
        s = s[0:width-3] + "..."
    return s

def makeHeaderBar(width,hbar,lc,rc):
    header = lc + hbar * width + rc + "\n"
    return header

def addTitle(header,title,indent,spaces):
    if title == None:
        return
    
    if len(title) != 0:
        title_chars     = len(title) + 2*spaces
        # indx            = slice(indent+1,indent+1+title_chars) 
        title_str = " "*spaces + title + " "*spaces 
        header_new  =  header[0:indent+1] + title_str + header[indent+1+title_chars:]
        return header_new
    else:
        return header

def getHeaders(ascii,header_width):

    if ascii:
        [hbar,vbar,lt,rt,lb,rb] = getSymbolsASCII()
    else:
        [hbar,vbar,lt,rt,lb,rb] = getSymbolsExtended()
    

    # set up top and bottom
    top             = makeHeaderBar(header_width,hbar,lt,rt)
    bottom          = makeHeaderBar(header_width,hbar,lb,rb)

    return [top,bottom,vbar]

def lineFormatStr(max_line_width,margin_spaces,vbar=None):
    margin          = " "*margin_spaces
    if vbar == None:
        format_str  = " " + margin + '%s\n'
    else:
        vbar        = vbar[0]
        s_chars     = str(max_line_width)
        format_str  = vbar + margin +  '%-' + s_chars + 's' + margin + vbar + '\n'

    return format_str

def getSymbolsExtended():
    hbar    = "═"
    vbar    = "║"
    lt      = "╔" 
    rt      = "╗"  
    lb      = "╚" 
    rb      = "╝"
    # hbar    = str(int("2550",16))  
    # vbar    = str(int("2551",16))   
    # lt      = str(int("2554",16))  
    # rt      = str(int("2557",16))  
    # lb      = str(int("255A",16))  
    # rb      = str(int("255D",16))  
    return [hbar,vbar,lt,rt,lb,rb]

def getSymbolsASCII():
    return [chr(35)]*6


        
    
