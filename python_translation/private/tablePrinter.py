from operator import imod
from private.linesearchWeakWolfe import linesearchWeakWolfe

from numpy.core.defchararray import isspace
from pygransoStruct import genral_struct
from private import formatOrange as fO, centerString as cS

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
   processed = "HelloWorld"
   print("TODO: tablePrinter.processLabel")
#  processed       = cellfun(  @(x) truncate(x,width), splitted,   ...
#                              'UniformOutput',        false       );
#  if add_arrows and ~isempty(strtrim(label)): 
#      process_fn  = @(x) addArrows(x,width);
#  else
#      process_fn  = @(x) centerString(x,width);
#  end
#  processed       = cellfun(  process_fn,         processed,      ...
#                              'UniformOutput',    false           );
   n               = len(processed) 
   return [processed,n]

def getEmptyStrings(n):
   c = np.empty((n,1),dtype=object)
   # if n > 0: 
   #   c = cellfun(@(x) '', c,'UniformOutput',false);
   
   return c

def spanLabel(label,lines):
   spannedLabel    = getEmptyStrings(lines - len(label)) + label[:]
   return spannedLabel

def processLabels(labels,widths,add_arrows):
   # [labels,lines]  = cellfun(  @(l,w) processLabel(l,w,add_arrows),    ...
   #                            labels, num2cell(widths),               ...
   #                            'UniformOutput',false                   );
   # labels          = cellfun(  @(x) spanLabel(x,max(cell2mat(lines))), ...
   #                            labels,                                 ...
   #                            'UniformOutput',false                   );
   # label_array     = horzcat(labels{:});
   print("TODO: ")
   label_array = "TODO: tablePrinter.processLabels"
   return label_array

def getTotalWidth(widths,delim_width,col_start,col_end):
   w = sum(widths[col_start:col_end]) + delim_width*(col_end-col_start)
   return w

def parseSpanningLabels(spanning_labels):
   # [labels,spans,col_starts] = cellfun(@parseSpanningLabel,        ...
   #                                      spanning_labels,            ...
   # #                                      'UniformOutput',false       );
   # return [labels,spans,col_starts]
   return [-1,-1,-1]

def parseSpanningLabel(spanning_label):
   label   = spanning_label[0]
   start   = spanning_label[1]
   span    = start + spanning_label[2]
   return [label,span,start]

def moveSpanLabels(span_labels,indx,n_cols):
   labels = np.empty(n_cols,dtype=object)
   for i in range(n_cols):
      labels[i] = ""

   labels(indx)        = span_labels
   return labels

def processSpannedLabels(   labels, widths, span_labels, row_str, vs  ):
                                                            
   [span_labels,spans] = parseSpanningLabels(span_labels)
   
   delim_width         = len(vs)
   [n_lines,n_cols]    = len(labels)
   n_multicols         = len(spans)
   
   del_indx            = np.zeros((1,n_cols))
   span_indx           = np.zeros((1,n_cols))
   
   for j in range(n_multicols):
      span        = spans[j]
      col_start   = span[0]
      col_end     = span[1]
      span        = slice(col_start,col_end)
         
      #  join column labels for multicolumns and always do joins for the
      #  the very last line
      for k in range(n_lines):
         labels_to_join          = labels[k,span]
         if k == n_lines or not allEmpty(labels_to_join):
               labels[k,col_start] = vs.join(labels_to_join) 
         
      span_indx[col_start]    = True
      widths[col_start]       = getTotalWidth( widths, delim_width, col_start, col_end )
      row_str[col_start]      = vs.join(row_str(span))   
      del_indx[span[1:]]   = True 
   
   #  delete columns that have been joined into first column of each span
   labels[:,del_indx]  = None
   widths[del_indx]    = None
   row_str[del_indx]   = None
   span_indx[del_indx] = None
   
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
   indx = [np.any(not isspace(label)) for label in labels]  
   return indx

def findOverlap(labels,spanned_indx):
   lines   = labels.shape[0]  
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
   [n_top,n_cols]  = labels_top.shape
   n_bottom        = labels_bottom.shape[0]
   n_lines         = n_top + n_bottom - overlapping_lines
   
   labels          = np.empty([n_lines,n_cols], dtype="S10")  

   indx_bottom     = getLabelLocations(labels_bottom)
   indx            = np.vstack(np.zeros((n_lines-n_bottom,n_cols)), indx_bottom)  
   labels[indx]    = labels_bottom[indx_bottom]
   
   indx_top        = getLabelLocations(labels_top)
   indx            = np.vstack(indx_top, np.zeros((n_lines-n_top,n_cols)))
   labels[indx]    = labels_top[indx_top]

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

   [vs,vd]                 = getDelimitersVertical(use_ascii,spacing)
   
   n_cols                  = len(labels)
   labels                  = processLabels(labels,widths,False)
   row_str = np.empty((1,n_cols),dtype="S10")
   row_str[:] = "%s"
   
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
      
      label =  "<" + "-"*arrow_length_left + label + "-" * arrow_length_right   
   else:
      label = cS.centerString(label,width)
   
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