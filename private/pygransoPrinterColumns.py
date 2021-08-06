from pygransoStruct import general_struct
from private import nDigitsInWholePart as nDIWP, truncate, centerString as cS, double2FixedWidthStr as d2FWS, formatOrange as fO

def pygransoPrinterColumns(opts,ineq_constraints,eq_constraints):
    #    gransoPrinterColumns:
    #        Sets up formatters for each column needed for GRANSO's printer,
    #        gransoPrinter.

    field_width     = min(max(9,opts.print_width),23)
    constrained     = ineq_constraints or eq_constraints
   
    #  set up all the column formatters
    c = general_struct()
    setattr(c,"iter", getCount("Iter",opts.maxit))
    
    if constrained:
        setattr(c,"mu", getNumberColumn("Mu",9,True))
        setattr(c,"pen", getNumberColumn("Value",field_width))
    else:
        setattr(c,"mu", getBlank("Mu"))
        setattr(c,"pen", getBlank("Value"))
    
    setattr(c,"obj", getNumberColumn("Objective",field_width))
    setattr(c,"ineq", violationFormatter("Ineq",ineq_constraints))
    setattr(c,"eq", violationFormatter("Eq",eq_constraints))
 
    gSD_object = gSD()
    setattr(c,"sd", gSD_object.getSearchDirection(constrained,opts))
    setattr(c,"ls_evals", getCount("Evals",opts.ls_max_estimate))
    setattr(c,"ls_size", getNumberColumn("t",9,True))
    setattr(c,"ngrad", getCount("Grads",opts.ngrad))
    gSM_object = gSM()
    setattr(c,"stat_value", gSM_object.getStationarityMeasure(opts.use_orange, opts.use_ascii))
    
    return c

def getCount(label,maxit):
    width   = max(nDIWP.nDigitsInWholePart(maxit),len(label))
    iter = general_struct()
    setattr(iter,"label",label)
    setattr(iter,"width",width)
    setattr(iter,"format_fn",lambda x: "%*s" % (width,x) )
    # setattr(iter,"format_str",lambda x: " "*(width-1) + truncate.truncate(x,width) )
    setattr(iter,"format_str",lambda x: "%*s" % (width,truncate.truncate(x,width))  )
    setattr(iter,"blank_str"," " * width)
    setattr(iter,"na_str",cS.centerString('-',width) )
    setattr(iter,"label",label)
    return iter

def getBlank(label,user_width = -1):   
    label = label.strip()
    width = len(label)
    width = max(width,user_width)
    type = general_struct()
    setattr(type,"label",label)
    setattr(type,"width",width)
    setattr(type,"blank_str"," " * width)
    setattr(type,"dash_str",cS.centerString('-',width))
    setattr(type,"na_str",cS.centerString('n/a',width))
    return type

def getNumberColumn(label,num_width,nonnegative=False):
    get_num_fn              = d2FWS.double2FixedWidthStr(num_width)
    if nonnegative:
        format_number_fn    = lambda x: dropFirstChar(get_num_fn(x))
        num_width           = num_width - 1
    else:
        format_number_fn    = get_num_fn

    type = general_struct()
    setattr(type,"label",label)
    setattr(type,"width",num_width)
    setattr(type,"format_fn",format_number_fn)
    setattr(type,"blank_str"," " * num_width)
    setattr(type,"dash_str",cS.centerString('-',num_width))
    setattr(type,"na_str",cS.centerString('n/a',num_width))
    
    return type

class gSD:
    def __init__(self):
        pass
    
    def getSearchDirection(self,constrained,opts):
        self.constrained = constrained
        random_attempts = opts.random_attempts
        
        if opts.use_orange and not opts.use_ascii:
            format_fn = lambda sd_code,random_attempts: self.formatSearchDirectionOrange(sd_code,random_attempts,constrained)
        else:
            format_fn = lambda sd_code,random_attempts: self.formatSearchDirection(sd_code,random_attempts)
        
        if random_attempts > 0:
            self.width = max(nDIWP.nDigitsInWholePart(random_attempts)+1, 2)
        else:
            self.width = 2
        
        self.sd_str_fn = lambda code,tries: searchDirectionStr(code,tries,self.width)
        
        type = general_struct()
        setattr(type,"label","SD")
        setattr(type,"width",self.width)
        setattr(type,"format_fn",format_fn)
        setattr(type,"blank_str"," " * self.width)
        setattr(type,"na_str",cS.centerString('-',self.width))

        return type
        
    def formatSearchDirection(self,sd_code,random_attempts):
        str = "%-*s"%(self.width,self.sd_str_fn(sd_code,random_attempts))
        return str
        
    def formatSearchDirectionOrange(self,sd_code,random_attempts,constrained):
        str = self.formatSearchDirection(sd_code,random_attempts)
        
        #  print search direction type in orange if it isn't the default
        if sd_code > 2 or (self.constrained and sd_code > 0):
            str = fO.formatOrange(str)

        return str

class gSM:
    def __init__(self):
        pass

    def getStationarityMeasure(self,use_orange,use_ascii):
        get_num_fn          = d2FWS.double2FixedWidthStr(9)
        self.format_number_fn    = lambda x: dropFirstChar(get_num_fn(x))
        width               = 10
        
        if use_orange and not use_ascii:
            format_fn       = lambda value,stat_type: self.formatStationarityMeasureOrange(value,stat_type)
        else:
            format_fn       = lambda value,stat_type: self.formatStationarityMeasure(value,stat_type)
        
        type = general_struct()
        setattr(type,"label","Value")
        setattr(type,"width",width)
        setattr(type,"format_fn",format_fn)
        setattr(type,"blank_str"," " * width)
        setattr(type,"dash_str",cS.centerString('-',width-2))
        setattr(type,"na_str",cS.centerString('n/a',width-2))

        return type
    
    def formatStationarityMeasure(self,value,stat_type):
        value = self.format_number_fn(value)
        if stat_type > 1:
            str = "%s:%d"%(value,stat_type) 
        else:
            str = "%s  "%(value) 
        return str

    def formatStationarityMeasureOrange(self,value,stat_type):
        value = self.format_number_fn(value)
        if stat_type > 1:
            str = "%s:%d"%(value,stat_type)  
            str = fO.formatOrange(str)
        else:
            str = "%s  "%(value)
        return str

def violationFormatter(label,n_constraints):
    if n_constraints > 0:
       col = getNumberColumn(label,9,True)
    else:
       col = getBlank(label,4)
    return col

def dropFirstChar(s):
    s = s[1:]
    return s

def searchDirectionStr(sd_code,random_attempts,width):
    if sd_code == 0:      # Steering
        s = "S"
    elif sd_code == 1:    # Steering w/ I in lieu of inverse Hessian approx
        s = "SI"
    elif sd_code == 2:    # Regular BFGS on penalty function
        s = "QN"
    elif sd_code == 3:    # Gradient descent on penalty function
        s = "GD"
    elif sd_code == 4:    # Random search direction
        s = "R%0*d"%(width-1,random_attempts)
    else:                 # not applicable (e.g. first iteration)
        s = "-"
    return s

