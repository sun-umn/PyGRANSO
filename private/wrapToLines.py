import numpy as np

def wrapToLines(str_in,width,indent):
    """
    wrapToLines:
      Converts the string str_in into a cell array of lines, each of at most
      length width by doing basic word wrapping (no hyphenation).  Each
      line can be optionally indented.  Words will be truncated if their 
      length exceeds the specified width minus the indent value.
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