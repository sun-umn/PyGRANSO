from private import printOrange as pO

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

def printMessageBox(use_ascii,use_orange,margin_spaces,title_top,title_bottom,msg_lines,sides=True,user_width=0):
    # printMessageBox:
    #    Prints a message to the console surrounded by a border.
    #    If user_width is not provided, the border is automatically sized to
    #    fit the maximum width line (including given whitespace formatting)
    #    specified in the message.

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
        
    
