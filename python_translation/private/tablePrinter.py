from operator import imod
from os import truncate

from numpy.lib.function_base import append
from private.linesearchWeakWolfe import linesearchWeakWolfe

from numpy.core.defchararray import isspace
from pygransoStruct import genral_struct
from private import formatOrange as fO, truncate as TR
from private.centerString import centerString

from itertools import compress

import numpy as np
import math, string

class tP:
   def __init__(self):
       pass

   def tablePrinter( self, use_ascii,  use_orange, labels, widths, spacing, span_labels = None ):
      """   
      tablePrinter:
         An object that makes it much easier to print nice looking tables to
         Python's console.  
      """

      print("TODO: private.tablePrinter")

      self.use_orange = use_orange
      
      if np.all(self.use_orange==None):
         self.use_orange  = False
      
      
      [self.header,self.rules,self.row_str,self.msg_str,self.msg_width] = init(use_ascii, labels, widths, spacing, span_labels) 
                                                      
      [self.printOverlayFn,overlay_width]  = overlayPrinter(self.rules.m.rs,self.msg_width)
                                                   
      self.last_printed = "n"
   
      printer = genral_struct()
      setattr(printer,"header", lambda : printHeader())
      setattr(printer,"row", lambda : printRow())
      setattr(printer,"msg", lambda s : printMessage(False,s))
      setattr(printer,"msgOrange", lambda s: printMessage(self.use_orange,s))
      setattr(printer,"overlay", lambda s : printOverlay(False,s))
      setattr(printer,"overlayOrange", lambda s: printOverlay(self.use_orange,s))
      setattr(printer,"msgWidth", lambda : self.msg_width())
      setattr(printer,"overlayWidth", lambda : overlay_width())
      setattr(printer,"close", lambda : printClose())
      
                              
   def printHeader(self):
      if self.last_printed == "n":
         r = self.rules.t.h
      elif self.last_printed == "m":
         r = self.rules.m.fh
      if self.last_printed == "r" or self.last_printed == "h" or self.last_printed == "w":
         r = self.rules.m.rh
             
      print(r)
      print(self.header)
      
      self.last_printed = "h"
   

   def printRow(self,varargin):
      if self.last_printed == "n":
         r = self.rules.t.r
      elif self.last_printed == "m":
         r = self.rules.m.fr
      elif self.last_printed == "h" or self.last_printed == "w":
         r = self.rules.m.r
      
      if self.last_printed != "r":
         print(r)
      
      
      print(self.row_str + varargin)
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
      
      
      print(r)
      if np.ischar(msg_lines):
            print("TODO: tablePrinter np.ischar(msg_lines)")
            msg_lines   = msg_lines
      
      if use_orange: 
            msg_line_fn = lambda : messageLineOrange()
      else:
            msg_line_fn = lambda : messageLineBlack()
      
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


         print(r)
         self.printOverlayFn(use_orange,msg_lines)
         self.last_printed    = "w"
      

      def printClose(self):

         if self.last_printed == "m":
            r = self.rules.b.f
         elif self.last_printed == "r" or self.last_printed == "h" or self.last_printed == "w":
            r = self.rules.b.r


         if self.last_printed != "n":
               print(r)
               self.last_printed = "n"
         

      def messageLineBlack(self,str):
         str = prepareMessageStr(self.msg_width,self.msg_width,str)
         print(self.msg_str,str)
         

      def messageLineOrange(str):
         str = prepareMessageStr(self.msg_width,self.msg_width,str)
         if self.use_orange and len(str) > 0:
               str = fO.formatOrange(str)
         
         print(self.msg_str,str)
      


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
         
         print(self.rule[0:3] + "   "  + str + "   " + self.rule[self.max_width + 10:])
        
   

def prepareMessageStr(print_width,max_width,str):
    chars   = len(str)
    if print_width < chars:
        str = truncateStr(str,print_width)
    else: # the string is not long enough so pad right side
        str = "%-*s"%(max_width,str) 
    return str

def processLabel(label,width,add_arrows):
   label           = label
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
   print("TODO: get getEmptyStrings")
   c = np.empty((n,1),dtype=object)
   # if n > 0: 
   #   c = cellfun(@(x) '', c,'UniformOutput',false);
   
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
   w = sum(widths[col_start:col_end]) + delim_width*(col_end-col_start)
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
   # [n_lines,n_cols]    = len(labels)
   n_cols = len(labels)
   n_lines = 1
   print("HARDCODE: processSpannedLabels")

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
      widths_list[col_start]       = getTotalWidth( widths, delim_width, col_start, col_end )
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
      vs  = "|"
      vd  = "|"
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
   
   row_str = []
   for i in range(n_cols):
      row_str.append("%s")

   if np.any(span_labels != None):
      [labels,row_str]    = processSpannedLabels( labels, widths, span_labels, row_str, vs )
   
   
   indx            = getLabelLocations(labels)
   widths          = [len(label) for label in labels[-1,:]]  
   blank_strs      = [" " * width for width in widths]  
   blank_strs      = np.tile(blank_strs,labels.shape[0])
   labels[not indx]   = blank_strs[not indx] 
   
   #  transpose output to a row cell array since mat2cell outputs to a
   #  column and strjoin on 2014a will only accept rows, while on 2015a,
   #  strjoin can handle either.

   print("TODO: tablePrinter mat2cell( ,ones(1,size(labels,1)) ).'")
   # lines           = mat2cell( ,ones(1,size(labels,1)) ).';
   lines           = labels

   header_strs     = [formatLine(line,vd) for line in lines] 
   header_last     = header_strs[-1,0].rstrip()  
   header          = "".join(header_strs)                  
   row_str         = formatLine(row_str,vd)
   
   rules           = makeRules(use_ascii,header_last)
   msg_width       = len(header_last) - 2
   msg_str         = "%-" + str(msg_width) + "s" + vd + "\n"  

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
      
      label =  "<" + "-"*int(arrow_length_left) + label + "-" * int(arrow_length_right)   
   else:
      label = centerString(label,width)
   
   return label

def formatLine(formatted_labels,vd):
   line = vd.join(formatted_labels + "\n")   
   return line

def makeRules(use_ascii,h2):

   [vs,vd]             = getSymbolsVertical(use_ascii)
   get_symbols_fn      = lambda : getSymbolsASCII() if use_ascii else lambda: getSymbols()  
   [ hs, hd, cs, cd, csd, cds, tsdl, tdl, tds, td, tdsu, tdu, edtr, edbr ] = get_symbols_fn()
   
   vd_indx             = h2.find(vd)  
   vd_indx             = vd_indx[0:-2]
   vs_indx             = h2.find(vs)
   
   flat_m1             = hd * (len(h2) - 1)  

   # top rules, with downward corner piece at the end
   top_f               = flat_m1 + edtr + "\n"  # top rule - flat 
   
   top_h               = top_f                # top rule - header
   top_h[vd_indx]      = td
   
   top_r               = top_h                # top rule - row
   top_r[vs_indx]      = tds
   
   #  bottom rules, with upward corner piece at the end
   bottom_f            = flat_m1 + edbr + "\n"  # bottom rule - flat
   
   bottom_r            = bottom_f
   bottom_r[vd_indx]   = tdu                  # bottom rule - header/row
   bottom_r[vs_indx]   = tdsu
   
   #  mid rules, with left-pointing T piece at the end 
   mid_f               = flat_m1 + tdl + "\n"   # mid rule - flat to flat
   
   mid_fh              = mid_f                # mid rule - flat to header
   mid_fh[vd_indx]     = td
   
   mid_fr              = mid_fh               # mid rule - flat to row
   mid_fr[vs_indx]     = tds
   
   mid_r               = mid_f                # mid rule - row to row
   mid_r[vd_indx]      = cd                   # (also header to row)
   mid_r[vs_indx]      = cds
   
   mid_rs              = string.replace(mid_r,hd,hs)  # mid rule - row to row
   mid_rs[vd_indx]     = csd                  # single line, not double
   mid_rs[vs_indx]     = cs
   mid_rs              = string.replace(mid_rs,tdl,tsdl)
   
   mid_rh              = mid_f                # mid rule - row to header
   mid_rh[vd_indx]     = cd
   mid_rh[vs_indx]     = tdsu
   
   mid_rf              = mid_f                # mid rule - row to flat
   mid_rf[vd_indx]     = tdu                  # (also header to flat)
   mid_rf[vs_indx]     = tdsu
   
   t = genral_struct()
   setattr(t, "f", top_f)
   setattr(t, "h", top_h)
   setattr(t, "r", top_r)

   b = genral_struct()
   setattr(b, "f", bottom_f)
   setattr(b, "r", bottom_r) 
   
   m = genral_struct()
   setattr(m, "f", mid_f)
   setattr(m, "fh", mid_fh)
   setattr(m, "fr", mid_fr)
   setattr(m, "r", mid_r)
   setattr(m, "rh", mid_rh)
   setattr(m, "rf", mid_rf)
   setattr(m, "rs", mid_rs)

   rules = genral_struct()     
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

   print("private.tablePrinter: double and single quote inconsistent")
   # double and single quote inconsistent
   tsdl                = '╢'
   tdl                 = '╣'
   tds                 = '╤'
   td                  = '╦'
   tdsu                = '╧'
   tdu                 = '╩'

   edtr                = '╗'
   edbr                = '╝'

   return [  hs, hd, cs, cd, csd, cds, tsdl, tdl, tds, td, tdsu, tdu, edtr, edbr ]

def getSymbolsASCII():
      
   hs                  = '-'
   hd                  = '='

   cs                  = '|'
   cd                  = '|'
   csd                 = '|'
   cds                 = '|'

   tsdl                = '-'
   tdl                 = '='
   tds                 = '='
   td                  = '='
   tdsu                = '='
   tdu                 = '='

   edtr                = '='
   edbr                = '='

   return [  hs, hd, cs, cd, csd, cds, tsdl, tdl, tds, td, tdsu, tdu, edtr, edbr ]