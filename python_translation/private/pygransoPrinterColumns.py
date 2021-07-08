from pygransoStruct import genral_struct
from private import nDigitsInWholePart as nDIWP, truncate, centerString as cS, double2FixedWidthStr as d2FWS, formatOrange as fO

def getCount(label,maxit):
    width   = max(nDIWP.nDigitsInWholePart(maxit),len(label))
    iter = genral_struct()
    setattr(iter,"label",label)
    setattr(iter,"width",width)
    setattr(iter,"format_fn",lambda x: " "*(width-1) + str(x) )
    setattr(iter,"format_str",lambda x: " "*(width-1) + truncate.truncate(x,width) )
    setattr(iter,"blank_str"," " * width)
    setattr(iter,"na_str",cS.centerString('-',width) )
    setattr(iter,"label",label)
    return iter

def getBlank(label,user_width = -1):   
    label = label.strip()
    width = len(label)
    width = max(width,user_width)
    type = genral_struct()
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
   
    type = genral_struct()
    setattr(type,"label",label)
    setattr(type,"width",num_width)
    setattr(type,"format_fn",format_number_fn)
    setattr(type,"blank_str"," " * num_width)
    setattr(type,"dash_str",cS.centerString('-',num_width))
    setattr(type,"na_str",cS.centerString('n/a',num_width))
    
    return type


def getSearchDirection(constrained,opts):
    random_attempts = opts.random_attempts
    
    if opts.use_orange:
        format_fn = lambda sd_code,random_attempts: formatSearchDirectionOrange(sd_code,random_attempts,constrained)
    else:
        format_fn = lambda sd_code,random_attempts: formatSearchDirection(sd_code,random_attempts)
    
    if random_attempts > 0:
        width = max(nDIWP.nDigitsInWholePart(random_attempts)+1, 2)
    else:
        width = 2
    
    sd_str_fn = lambda code,tries: searchDirectionStr(code,tries,width)
    
    type = genral_struct()
    setattr(type,"label","SD")
    setattr(type,"width",width)
    setattr(type,"format_fn",format_fn)
    setattr(type,"blank_str"," " * width)
    setattr(type,"na_str",cS.centerString('-',width))

    return type
    
def formatSearchDirection(sd_code,random_attempts):
    str = "%-*s"%(width,sd_str_fn(sd_code,random_attempts))
    return str
    
def formatSearchDirectionOrange(sd_code,random_attempts,constrained):
    str = formatSearchDirection(sd_code,random_attempts)
    
    #  print search direction type in orange if it isn't the default
    if sd_code > 2 or (constrained and sd_code > 0):
        str = fO.formatOrange(str)

    return str

def getStationarityMeasure(use_orange):
    get_num_fn          = double2FixedWidthStr(9);
    format_number_fn    = @(x) dropFirstChar(get_num_fn(x));
    width               = 10;
    
    if use_orange
        format_fn       = @formatStationarityMeasureOrange;
    else
        format_fn       = @formatStationarityMeasure;
    end
    
    type = struct(                                                      ...
        'label',                'Value',                                ... 
        'width',                width,                                  ...
        'format_fn',            format_fn,                              ...
        'blank_str',            blanks(width),                          ...
        'dash_str',             [centerString('-',width-2) '  '],       ...
        'na_str',               [centerString('n/a',width-2) '  ']      );

    return type
    
    function str = formatStationarityMeasure(value,stat_type)
        value = format_number_fn(value);
        if stat_type > 1
            str = sprintf('%s:%d',value,stat_type);
        else
            str = sprintf('%s  ',value);
        end
    end

    function str = formatStationarityMeasureOrange(value,stat_type)
        value = format_number_fn(value);
        if stat_type > 1
            str = sprintf('%s:%d',value,stat_type);
            str = formatOrange(str);
        else
            str = sprintf('%s  ',value);
        end
    end
end

function col = violationFormatter(label,n_constraints)
    if n_constraints > 0
       col = getNumberColumn(label,9,true);
    else
       col = getBlank(label,4);
    end
end

function s = dropFirstChar(s)
    s = s(2:end);
end

function s = searchDirectionStr(sd_code,random_attempts,width)
    switch sd_code
        case 0          % Steering
            s = 'S';   
        case 1          % Steering w/ I in lieu of inverse Hessian approx
            s = 'SI';  
        case 2          % Regular BFGS on penalty function
            s = 'QN';   
        case 3          % Gradient descent on penalty function
            s = 'GD';   
        case 4          % Random search direction
            s = sprintf('R%0*d',width-1,random_attempts);
        otherwise       % not applicable (e.g. first iteration)
            s = '-';
    end
end

def gransoPrinterColumns(opts,ineq_constraints,eq_constraints):
    #    gransoPrinterColumns:
    #        Sets up formatters for each column needed for GRANSO's printer,
    #        gransoPrinter.

    field_width     = min(max(9,opts.print_width),23)
    constrained     = ineq_constraints or eq_constraints
   
    #  set up all the column formatters
    c = genral_struct()
    c.iter          = getCount('Iter',opts.maxit);
    
    if constrained
        c.mu        = getNumberColumn('Mu',9,true);
        c.pen       = getNumberColumn('Value',field_width);
    else
        c.mu        = getBlank('Mu');
        c.pen       = getBlank('Value');
    end
  
    c.obj           = getNumberColumn('Objective',field_width);
    c.ineq          = violationFormatter('Ineq',ineq_constraints);
    c.eq            = violationFormatter('Eq',eq_constraints);
 
    c.sd            = getSearchDirection(constrained,opts);
    c.ls_evals      = getCount('Evals',opts.ls_max_estimate);
    c.ls_size       = getNumberColumn('t',9,true);
    c.ngrad         = getCount('Grads',opts.ngrad);
    c.stat_value    = getStationarityMeasure(opts.use_orange);

    return c