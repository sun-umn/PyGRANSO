from numpy.core.defchararray import isspace
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private import formatOrange as fO, truncate as TR
from pygranso.private.centerString import centerString
from itertools import compress
import numpy as np
import math 

class tP:
    def __init__(self):
        pass

    def tablePrinter( self, use_ascii,  use_orange, labels, widths, spacing, span_labels = None ):
        """
        tablePrinter:
            An object that makes it much easier to print nice looking tables to
            Python's console.

            INPUT:
                use_ascii
                    Logical to indicate whether or not to use basic ascii
                    characters instead of the extended characters to print table
                    boundaries.

                use_orange
                    Logical to indicate whether or not to enable the orange
                    printing features; orange printing is an undocumented feature
                    in MATLAB hence the option to disable it if necessary.
                    As a shortcut, setting use_orange = [] is equivalent to false.

                labels
                    Cell array of strings, to label each column in the table.  New
                    lines are supported in these labels.

                widths
                    Array of positive integers, with each entry indicating the
                    printable width (in number of characters) of each column in the
                    table.  Labels will be truncated as necessary if their widths
                    exceed their corresponding column widths specified here.

                spacing
                    A nonnegative integer indicating the number of spaces to use as
                    left/right border margins for all entries.

                span_labels (optional)
                    A cell array of cell arrays to specify labels that should span
                    multiple columns.  Each entry must have the form:
                        {span_label_str,column_start,column_end}
                    Thus {'Group 1 data',2,4} would put span label 'Group 1 data'
                    centered above the regular labels for columns 2 through 4.
                    Overlapping span labels are NOT supported.

            OUTPUT:
                printer
                    A tablePrinter object with the following methods:

                    .header()
                        Prints a table header of all the column labels.  This can
                        be called multiple times, such as every twenty rows.

                    .row(col1_data,...,colk_data)
                        Prints a row in the table.  The number of provided input
                        arguments must match the number of columns in the table.
                        Each argument should be a string that represents the data
                        to be printed for each column.  Each string must have a
                        printable length that matches the width specified for its
                        given column (given by the input argument widths); if this
                        not done, or new lines appear in these strings, then the
                        alignment of the table will be broken.  Each input string
                        is processed as fprintf %s argument.

                    .msg(msg)
                        Prints a message inside the table, using its full width.
                        This function takes either a string or cell array, the
                        latter where each entry specifies a line in a multi-line
                        message.  Any line too long to be printed in the table will
                        be truncated.

                    .msgOrange(msg)
                        Same as .msg but will print the message using orange text.

                    .overlay(msg)
                        Prints a message overlaid on the table.  Though this
                        produces a different look of the message, this function
                        takes the same input arguments as .msg.

                    .overlayOrange(msg)
                        Same as .overlay but prints the overlay message using
                        orange text.

                    .msgWidth()
                        Get the number of printable characters for a single line
                        in a message printed using .msg.

                    .overlayWidth()
                        Get the number of printable characters for a single line
                        in a message printed using .overlay.  Note that overlay
                        messages have less available width than regular messages.

                    .close()
                        Close the current table, that is, print the bottom border.
                        Note that there is no need to explictly open/begin a table;
                        this is handled automatically.  Merely requesting something
                        is a printed (a header, a row, or a message) will print
                        also print the top border of a table as necessary, such as
                        if no printing has already been done or if .close() was
                        called to close a preceding table.

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

        tablePrinter.py (introduced in PyGRANSO v1.0.0)
        Copyright (C) 2016-2021 Tim Mitchell

        This file is a direct port of tablePrinter.m, which is included as part
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

        self.use_orange = use_orange
        self.use_ascii = use_ascii

        if np.all(self.use_orange==None):
            self.use_orange  = False


        [self.header,self.rules,self.row_str,self.msg_str,self.msg_width] = init(use_ascii, labels, widths, spacing, span_labels)

        oP = Class_overlayPrinter()
        [self.printOverlayFn,overlay_width]  = oP.overlayPrinter(self.rules.m.rs,self.msg_width)

        self.last_printed = "n"

        printer = pygransoStruct()
        setattr(printer,"header", lambda : self.printHeader())
        setattr(printer,"row", lambda varargin: self.printRow(varargin))
        setattr(printer,"msg", lambda s : self.printMessage(False,s))
        setattr(printer,"msgOrange", lambda s: self.printMessage(self.use_orange,s))
        setattr(printer,"overlay", lambda s : self.printOverlay(False,s))
        setattr(printer,"overlayOrange", lambda s: self.printOverlay(self.use_orange,s))
        setattr(printer,"msgWidth", lambda : self.msg_width )
        setattr(printer,"overlayWidth", lambda : overlay_width )
        setattr(printer,"close", lambda : self.printClose())

        return printer


    def printHeader(self):
        if self.last_printed == "n":
            r = self.rules.t.h
        elif self.last_printed == "m":
            r = self.rules.m.fh
        if self.last_printed == "r" or self.last_printed == "h" or self.last_printed == "w":
            r = self.rules.m.rh

        print(r,end="")
        print(self.header,end="")

        self.last_printed = "h"


    def printRow(self,varargin):
        if self.last_printed == "n":
            r = self.rules.t.r
        elif self.last_printed == "m":
            r = self.rules.m.fr
        elif self.last_printed == "h" or self.last_printed == "w":
            r = self.rules.m.r

        if self.last_printed != "r":
            print(r,end="")


        print(self.row_str % varargin,end="")
        self.last_printed = "r"


    def printMessage(self,use_orange,msg_lines):
        if len(msg_lines) < 1:
            return

        if self.last_printed == "n":
            r = self.rules.t.f
        elif self.last_printed == "m":
            r = self.rules.m.f
        elif self.last_printed == "r" or self.last_printed == "h" or self.last_printed == "w":
            r = self.rules.m.rf


        print(r,end="")


        if use_orange:
                msg_line_fn = lambda s: self.messageLineOrange(s)
        else:
                msg_line_fn = lambda s: self.messageLineBlack(s)

        for msg_line in msg_lines:
            msg_line_fn(msg_line)

        self.last_printed    = "m"


    def printOverlay(self,use_orange,msg_lines):
        if len(msg_lines) < 1:
                return
        if self.last_printed == "n":
            r = self.rules.t.r
        elif self.last_printed == "m":
            r = self.rules.m.fr
        elif self.last_printed == "r" or self.last_printed == "h" or self.last_printed == "w":
            r = self.rules.m.r


        print(r,end="")
        self.printOverlayFn(use_orange,msg_lines)
        self.last_printed    = "w"


    def printClose(self):

        if self.last_printed == "m":
            r = self.rules.b.f
        elif self.last_printed == "r" or self.last_printed == "h" or self.last_printed == "w":
            r = self.rules.b.r


        if self.last_printed != "n":
                print(r,end="")
                self.last_printed = "n"


    def messageLineBlack(self,str):
        str = prepareMessageStr(self.msg_width,self.msg_width,str)
        print(self.msg_str % str,end="")


    def messageLineOrange(self,str):
        str = prepareMessageStr(self.msg_width,self.msg_width,str)
        if self.use_orange and len(str) > 0 and not self.use_ascii:
                str = fO.formatOrange(str)

        print(self.msg_str % str,end="")



######################################################################
######################################################################

class Class_overlayPrinter:

    def overlayPrinter(self,rule,msg_width):
        # msg_width minus the first 3 chars of the rule, and then 2*3 chars
        # for 3 space margins on each side of a message line
        self.rule = rule
        self.overlay_width       = msg_width - 9
        print_overlay_fn    = lambda : self.printOverlay()
        return [print_overlay_fn, self.overlay_width]


    def printOverlay(self,use_orange,msg_lines):
        self.use_orange = use_orange
        if np.ischar(msg_lines):
                msg_lines   = msg_lines

        self.max_width       = np.max([len(msg_line) for msg_line in msg_lines])

        for msg_line in msg_lines:
            self.printOverlayLine(msg_line)

    def printOverlayLine(self,str):
            str = prepareMessageStr(self.overlay_width,self.max_width,str)
            if self.use_orange and len(str) > 0:
                str = fO.formatOrange(str)

            print(self.rule[0:3] + "   "  + str + "   " + self.rule[self.max_width + 10:],end="")



def prepareMessageStr(print_width,max_width,str):
    chars   = len(str)
    if print_width < chars:
        str = truncateStr(str,print_width)
    else: # the string is not long enough so pad right side
        str = "%-*s"%(max_width,str)
    return str

def processLabel(label,width,add_arrows):
    # label           = label
    splitted        = label.split("\n")

    processed_tmp = []
    for splitted_element in splitted:
        processed_tmp.append(TR.truncate(splitted_element,width))

    if add_arrows and len(label.strip()) > 0:
        process_fn  = lambda x: addArrows(x,width)
    else:
        process_fn  = lambda x: centerString(x,width)

    processed = []
    for processed_tmp_element in processed_tmp:
        processed.append(process_fn(processed_tmp_element))

    n               = len(processed)
    return [processed,n]

def getEmptyStrings(n):
    c = np.empty((n,1),dtype=object)

    return c

def spanLabel(label,lines):
   spannedLabel    =  label
   return spannedLabel

def processLabels(labels,widths,add_arrows):

    n = len(labels)
    labels_new = []
    lines_new = []
    for i in range(n):
        [label, line] = processLabel(labels[i], widths[i],add_arrows)
        labels_new.append(label)
        lines_new.append(line)

    labels = labels_new.copy()
    lines = lines_new.copy()
    labels_new = []

    for i in range(n):
        #  second [0] means get string from list of string
        label = spanLabel( labels[i][0],max(lines) )
        labels_new.append(label)

    label_array = labels_new.copy()


    return label_array

def getTotalWidth(widths,delim_width,col_start,col_end):
    w = sum(widths[col_start-1:col_end]) + delim_width*(col_end-col_start)
    return w

def parseSpanningLabels(spanning_labels):
    labels = []
    spans = []
    col_starts = []

    for spanning_label in spanning_labels:
        [label,span,col_start] = parseSpanningLabel(spanning_label)
        labels.append(label)
        spans.append(span)
        col_starts.append(col_start)

    return [labels,spans,col_starts]

def parseSpanningLabel(spanning_label):
    label   = spanning_label[0]
    start   = spanning_label[1]
    span    = (start , spanning_label[2])
    return [label,span,start]

def moveSpanLabels(span_labels,indx,n_cols):
    labels = []
    count = 0
    for i in range(n_cols):
        if indx[i]:
            labels.append(span_labels[count])
            count += 1
        else:
            labels.append("")

    return labels

def processSpannedLabels(   labels, widths, span_labels, row_str, vs  ):

    [span_labels,spans, *_] = parseSpanningLabels(span_labels)

    delim_width         = len(vs)
    n_cols = len(labels)
    n_lines = 1
    # spannLabel hardcode here

    n_multicols         = len(spans)

    del_indx            = np.ones(n_cols, dtype=int)
    span_indx           = np.zeros(n_cols, dtype=int)

    for j in range(n_multicols):
        span        = spans[j]
        col_start   = span[0] - 1 # python index -1
        col_end     = span[1]
        span        = slice(col_start,col_end)

        #  join column labels for multicolumns and always do joins for the
        #  the very last line
        for k in range(n_lines):
            # Hardcode n_lines to be 1 here
            labels_to_join          = labels[span]
            if k == n_lines or not allEmpty(labels_to_join):
                labels[col_start] = vs.join(labels_to_join)

        span_indx[col_start]    = True
        # 'tuple' object does not support item assignment
        widths_list = list(widths)
        widths_list[col_start]       = getTotalWidth( widths, delim_width, col_start+1, col_end )
        widths = widths_list
        row_str[col_start]      = vs.join(row_str[span])
        del_indx[col_start+1:col_end]   = False # we are deleting these indexs here

    #  delete columns that have been joined into first column of each span
    labels = list(compress(labels, del_indx))
    widths = list(compress(widths, del_indx))
    row_str = list(compress(row_str, del_indx))
    span_indx = list(compress(span_indx, del_indx))

    #  reset span_labels positions and format them
    n_cols              = len(widths)
    span_labels         = moveSpanLabels(span_labels,span_indx,n_cols)
    span_labels         = processLabels(span_labels,widths,True)

    #  merge both sets of labels together
    overlap_line        = findOverlap(labels,span_indx)
    labels              = mergeLabels(span_labels,labels,overlap_line)
    return [labels,row_str]

def allEmpty(cell_of_strs):
    s = " ".join(cell_of_strs).strip()
    tf = np.all(s==None)
    return tf

def getLabelLocations(labels):
    indx = []
    for label in labels:
        if len(label.strip()) == 0:
            indx.append(False)
        else:
            indx.append(True)

    return indx

def findOverlap(labels,spanned_indx):
    # Hardcode lines = 1 here
    lines   = 1
    line    = 0
    #  don't allow the last line to overlap
    for j in range(lines-1):
        line_labels     = labels[j,spanned_indx]
        nonempty_indx   = [np.any(not isspace(line_label)) for line_label in line_labels]
        if np.any(nonempty_indx!=0):
            break
        line = j
    return line

def mergeLabels(labels_top,labels_bottom,overlapping_lines):
    # hardcode nlines = 1
    [n_top,n_cols]  = [1,len(labels_top)]
    n_bottom        = 1
    n_lines         = n_top + n_bottom - overlapping_lines

    labels          = np.empty([n_lines,n_cols], dtype="S10")

    indx_bottom     = getLabelLocations(labels_bottom)

    # hardcode nlines = 1
    indx_top = []
    for i in range(n_cols):
        indx_top.append(False)
    indx            = [indx_top, indx_bottom]


    labels_bottom_new    = []
    labels_top_new = []
    indx_top        = getLabelLocations(labels_top)

    for i in range(n_cols):
        if indx_bottom[i]:
            labels_bottom_new.append(labels_bottom[i])
        else:
            labels_bottom_new.append("")

        if indx_top[i]:
            labels_top_new.append(labels_top[i])
        else:
            labels_top_new.append("")


    labels   = [labels_top_new, labels_bottom_new]

    return labels

def getSymbolsVertical(use_ascii):
    if use_ascii:
        vs  = chr(124)
        vd  = chr(124)
    else:
        vs  = "│"
        vd  = "║"
    return [vs,vd]

def getDelimitersVertical(use_ascii,spacing):
    [vs,vd] = getSymbolsVertical(use_ascii)
    space   = " " * spacing
    vs      = space + vs + space
    vd      = space + vd + space
    width   = len(vd)
    return [vs,vd,width]

def init( use_ascii, labels, widths, spacing, span_labels ):

    [vs,vd, *_]                 = getDelimitersVertical(use_ascii,spacing)

    n_cols                  = len(labels)
    labels                  = processLabels(labels,widths,False)

    row_str = ["%s"] * n_cols

    if np.any(span_labels != None):
        [labels,row_str]    = processSpannedLabels( labels, widths, span_labels, row_str, vs )



    indx_top        = getLabelLocations(labels[0])
    indx_bottom        = getLabelLocations(labels[1])
    indx            =  [indx_top, indx_bottom]

    widths          = [len(label) for label in labels[1]]
    blank_strs      = [" " * width for width in widths]

    # Hardcode label rows = 2 here
    blank_strs.extend(blank_strs)
    blank_strs      = [blank_strs, blank_strs]


    for i in range(len(labels[0])):
        # first row
        if not indx[0][i]:
            labels[0][i] = blank_strs[0][i]
        # second row
        if not indx[1][i]:
            labels[1][i] = blank_strs[1][i]



    #  transpose output to a row cell array since mat2cell outputs to a
    #  column and strjoin on 2014a will only accept rows, while on 2015a,
    #  strjoin can handle either.

    # print("TODO: tablePrinter mat2cell( ,ones(1,size(labels,1)) ).'")
    # lines           = mat2cell( ,ones(1,size(labels,1)) ).';
    lines           = labels

    header_strs     = [formatLine(line,vd) for line in lines]
    header_last     = header_strs[0].rstrip()
    header          = "".join(header_strs)
    row_str         = formatLine(row_str,vd)

    rules           = makeRules(use_ascii,header_last)
    msg_width       = len(header_last) - 2
    msg_str         = "%" + str(msg_width) + "s"  + vd + "\n"

    return [header,rules,row_str,msg_str,msg_width]

def addArrows(label,width):
    label       = label.strip()
    freespaces  = width - len(label)
    if freespaces < 0:
        label   = label[0:width]
    elif freespaces > 5:
        arrow_length_left       = (freespaces - 4)/2
        arrow_length_right      = arrow_length_left
        if freespaces % 2 > 0:
            arrow_length_left   = math.ceil(arrow_length_left)
            arrow_length_right  = math.floor(arrow_length_right)

        label =  "<" + "-"*int(arrow_length_left) + " " + label + " " + "-" * int(arrow_length_right) + ">"
    else:
        label = centerString(label,width)

    return label

def formatLine(formatted_labels,vd):
    line = vd.join(formatted_labels) + vd + "\n"
    return line

def makeRules(use_ascii,h2):

    [vs,vd]             = getSymbolsVertical(use_ascii)
    # get_symbols_fn      = lambda : getSymbolsASCII() if use_ascii else lambda: getSymbols()
    # [ hs, hd, cs, cd, csd, cds, tsdl, tdl, tds, td, tdsu, tdu, edtr, edbr ] = get_symbols_fn()

    if use_ascii:
        [ hs, hd, cs, cd, csd, cds, tsdl, tdl, tds, td, tdsu, tdu, edtr, edbr ] = getSymbolsASCII()
    else:
        [ hs, hd, cs, cd, csd, cds, tsdl, tdl, tds, td, tdsu, tdu, edtr, edbr ] = getSymbols()



    vd_indx             = [pos for pos, char in enumerate(h2) if char == vd]
    vd_indx             = vd_indx[0:-1]

    vs_indx             = [pos for pos, char in enumerate(h2) if char == vs]

    flat_m1             = hd * (len(h2) - 1)

    # top rules, with downward corner piece at the end
    top_f               = flat_m1 + edtr + "\n"  # top rule - flat

    top_h               = top_f                # top rule - header
    top_h_list = list(top_h)
    for idx in vd_indx:
        top_h_list[idx]      = td
    top_h      = "".join(top_h_list)

    top_r               = top_h                # top rule - row
    top_r_list = list(top_r)
    for idx in vs_indx:
        top_r_list[idx]      = tds
    top_r = "".join(top_r_list)

    #  bottom rules, with upward corner piece at the end
    bottom_f            = flat_m1 + edbr + "\n"  # bottom rule - flat

    bottom_r            = bottom_f
    bottom_r_list = list(bottom_r)
    for idx in vd_indx:
        bottom_r_list[idx]   = tdu                  # bottom rule - header/row
    for idx in vs_indx:
        bottom_r_list[idx]   = tdsu
    bottom_r = "".join(bottom_r_list)

    #  mid rules, with left-pointing T piece at the end
    mid_f               = flat_m1 + tdl + "\n"   # mid rule - flat to flat

    mid_fh              = mid_f                # mid rule - flat to header
    mid_fh_list = list(mid_fh)
    for idx in vd_indx:
        mid_fh_list[idx]     = td
    mid_fh = "".join(mid_fh_list)

    mid_fr              = mid_fh               # mid rule - flat to row
    mid_fr_list = list(mid_fr)
    for idx in vs_indx:
        mid_fr_list[idx]     = tds
    mid_fr = "".join(mid_fr_list)

    mid_r               = mid_f                # mid rule - row to row
    mid_r_list = list(mid_r)
    for idx in vd_indx:
        mid_r_list[idx]      = cd                   # (also header to row)
    for idx in vs_indx:
        mid_r_list[idx]      = cds
    mid_r = "".join(mid_r_list)

    for mid_r_char in mid_r_list:
        if mid_r_char == hd:
            mid_r_char = hs
    mid_rs              = "".join(mid_r_list)  # mid rule - row to row
    mid_rs_list = list(mid_rs)
    for idx in vd_indx:
        mid_rs_list[idx]     = csd                  # single line, not double
    for idx in vs_indx:
        mid_rs_list[idx]     = cs
    for i in range(len(mid_rs_list)):
    # for mid_rs_char in mid_rs_list:
        if mid_rs_list[i] == tdl:
            mid_rs_list[i] = tsdl
        if mid_rs_list[i] == hd:
            mid_rs_list[i] = hs
    mid_rs              = "".join(mid_rs_list)

    mid_rh              = mid_f                # mid rule - row to header
    mid_rh_list = list(mid_rh)
    for idx in vd_indx:
        mid_rh_list[idx]     = cd
    for idx in vs_indx:
        mid_rh_list[idx]     = tdsu
    mid_rh = "".join(mid_rh_list)

    mid_rf              = mid_f                # mid rule - row to flat
    mid_rf_list = list(mid_rf)
    for idx in vd_indx:
        mid_rf_list[idx]     = tdu                  # (also header to flat)
    for idx in vs_indx:
        mid_rf_list[idx]     = tdsu
    mid_rf = "".join(mid_rf_list)

    t = pygransoStruct()
    setattr(t, "f", top_f)
    setattr(t, "h", top_h)
    setattr(t, "r", top_r)

    b = pygransoStruct()
    setattr(b, "f", bottom_f)
    setattr(b, "r", bottom_r)

    m = pygransoStruct()
    setattr(m, "f", mid_f)
    setattr(m, "fh", mid_fh)
    setattr(m, "fr", mid_fr)
    setattr(m, "r", mid_r)
    setattr(m, "rh", mid_rh)
    setattr(m, "rf", mid_rf)
    setattr(m, "rs", mid_rs)

    rules = pygransoStruct()
    setattr(rules, "t", t)
    setattr(rules, "b", b)
    setattr(rules, "m", m)

    return rules

def getSymbols():

    hs                  = "─"
    hd                  = "═"

    cs                  = "┼"
    cd                  = "╬"
    csd                 = "╫"
    cds                 = "╪"

    # double and single quote inconsistent
    tsdl                = "╢"
    tdl                 = "╣"
    tds                 = "╤"
    td                  = "╦"
    tdsu                = "╧"
    tdu                 = "╩"

    edtr                = "╗"
    edbr                = "╝"

    return [  hs, hd, cs, cd, csd, cds, tsdl, tdl, tds, td, tdsu, tdu, edtr, edbr ]

def getSymbolsASCII():

    hs                  = chr(45)
    hd                  = chr(61)

    cs                  = chr(124)
    cd                  = chr(124)
    csd                 = chr(124)
    cds                 = chr(124)

    tsdl                = chr(45)
    tdl                 = chr(61)
    tds                 = chr(61)
    td                  = chr(61)
    tdsu                = chr(61)
    tdu                 = chr(61)

    edtr                = chr(61)
    edbr                = chr(61)

    return [  hs, hd, cs, cd, csd, cds, tsdl, tdl, tds, td, tdsu, tdu, edtr, edbr ]
