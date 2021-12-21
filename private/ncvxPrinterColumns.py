from ncvxStruct import GeneralStruct
from private import nDigitsInWholePart as nDIWP, truncate, centerString as cS, double2FixedWidthStr as d2FWS, formatOrange as fO

def ncvxPrinterColumns(opts,ineq_constraints,eq_constraints):
    """       
    ncvxPrinterColumns:
        Sets up formatters for each column needed for NCVX's printer,
        ncvxPrinter.

        INPUT:
            opts    
                A struct of the necessary parameters:
            .use_orange         logical indicating whether or not to enable 
                                orange printing  
            .print_width        integer between 9 and 23 to indicate printing
                                widths of adjustable fields (values of the
                                penalty andd the objective functions)
            .maxit              max number of iterations
            .ls_max_estimate    estimate of the max number of line search 
                                evaluations that can ever be incurred
            .random_attempts    the max number of random search directions that
                                may ever be attempted in a single iteration
            .ngrad              the max number of gradients that are cached for
        
            ineq_constraints    logical or positive number indicating the 
                                presence of inequality constraints
        
            eq_constraints      logical or positive number indicating the 
                                presence of equality constraints 
        
        OUTPUT:
            A struct containing formatters for the following columns/fields:
            .iter               
            .mu
            .pen
            .obj
            .ineq
            .eq
            .sd
            .ls_evals
            .ls_size
            .ngrad
            .stat_value

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
            gransoPrinter.m introduced in GRANSO Version 1.0.

            Buyun Dec 20, 2021 (NCVX Version 1.0.0):
                ncvxPrinter.py is translated from gransoPrinter.m in GRANSO Version 1.6.4. 

        For comments/bug reports, please visit the NCVX webpage:
        https://github.com/sun-umn/NCVX
            
        NCVX Version 1.0.0, 2021, see AGPL license info below.

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

    field_width     = min(max(9,opts.print_width),23)
    constrained     = ineq_constraints or eq_constraints
   
    #  set up all the column formatters
    c = GeneralStruct()
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
    iter = GeneralStruct()
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
    type = GeneralStruct()
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

    type = GeneralStruct()
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
        
        type = GeneralStruct()
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
        
        type = GeneralStruct()
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

