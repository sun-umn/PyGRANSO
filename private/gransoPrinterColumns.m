function c = gransoPrinterColumns(opts,ineq_constraints,eq_constraints)
%   gransoPrinterColumns:
%       Sets up formatters for each column needed for GRANSO's printer,
%       gransoPrinter.
%       
%   INPUT:
%       opts    
%           A struct of the necessary parameters:
%       .use_orange         logical indicating whether or not to enable 
%                           orange printing  
%       .print_width        integer between 9 and 23 to indicate printing
%                           widths of adjustable fields (values of the
%                           penalty andd the objective functions)
%       .maxit              max number of iterations
%       .ls_max_estimate    estimate of the max number of line search 
%                           evaluations that can ever be incurred
%       .random_attempts    the max number of random search directions that
%                           may ever be attempted in a single iteration
%       .ngrad              the max number of gradients that are cached for
%
%       ineq_constraints    logical or positive number indicating the 
%                           presence of inequality constraints
%
%       eq_constraints      logical or positive number indicating the 
%                           presence of equality constraints 
%
%   OUTPUT:
%       A struct containing formatters for the following columns/fields:
%       .iter               
%       .mu
%       .pen
%       .obj
%       .ineq
%       .eq
%       .sd
%       .ls_evals
%       .ls_size
%       .ngrad
%       .stat_value
%       
%
%   If you publish work that uses or refers to GRANSO, please cite the 
%   following paper: 
%
%   [1] Frank E. Curtis, Tim Mitchell, and Michael L. Overton 
%       A BFGS-SQP method for nonsmooth, nonconvex, constrained 
%       optimization and its evaluation using relative minimization 
%       profiles, Optimization Methods and Software, 32(1):148-181, 2017.
%       Available at https://dx.doi.org/10.1080/10556788.2016.1208749
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   gransoPrinterColumns.m introduced in GRANSO Version 1.0.
%
% =========================================================================
% |  GRANSO: GRadient-based Algorithm for Non-Smooth Optimization         |
% |  Copyright (C) 2016 Tim Mitchell                                      |
% |                                                                       |
% |  This file is part of GRANSO.                                         |
% |                                                                       |
% |  GRANSO is free software: you can redistribute it and/or modify       |
% |  it under the terms of the GNU Affero General Public License as       |
% |  published by the Free Software Foundation, either version 3 of       |
% |  the License, or (at your option) any later version.                  |
% |                                                                       |
% |  GRANSO is distributed in the hope that it will be useful,            |
% |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
% |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
% |  GNU Affero General Public License for more details.                  |
% |                                                                       |
% |  You should have received a copy of the GNU Affero General Public     |
% |  License along with this program.  If not, see                        |
% |  <http://www.gnu.org/licenses/agpl.html>.                             |
% =========================================================================

    field_width     = min(max(9,opts.print_width),23);
    constrained     = ineq_constraints || eq_constraints;
   
    % set up all the column formatters
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
    
end

function iter = getCount(label,maxit)
    width   = max(nDigitsInWholePart(maxit),length(label));
    iter = struct(                                                      ...
        'label',        label,                                          ...
        'width',        width,                                          ...
        'format_fn',    @(x) sprintf('%*s',width,num2str(x)),           ...
        'format_str',   @(x) sprintf('%*s',width,truncate(x,width)),    ...
        'blank_str',    blanks(width),                                  ...
        'na_str',       centerString('-',width)                         );
end

function type = getBlank(label,user_width)   
    label = strtrim(label);
    width = length(label);
    if nargin > 1
        width = max(width,user_width);
    end
    type = struct(                                                      ...
        'label',                label,                                  ...
        'width',                width,                                  ...
        'blank_str',            blanks(width),                          ...
        'dash_str',             centerString('-',width),                ...
        'na_str',               centerString('n/a',width)               );
end

function type = getNumberColumn(label,num_width,nonnegative)
    get_num_fn              = double2FixedWidthStr(num_width);
    if nargin > 2 && nonnegative
        format_number_fn    = @(x) dropFirstChar(get_num_fn(x));
        num_width           = num_width - 1;
    else
        format_number_fn    = get_num_fn;
    end
   
    type = struct(                                                      ...
        'label',                label,                                  ...
        'width',                num_width,                              ...
        'format_fn',            format_number_fn,                       ...
        'blank_str',            blanks(num_width),                      ...
        'dash_str',             centerString('-',num_width),            ...
        'na_str',               centerString('n/a',num_width)           );
end

function type = getSearchDirection(constrained,opts)
    random_attempts = opts.random_attempts;
    
    if opts.use_orange
        format_fn = @formatSearchDirectionOrange;
    else
        format_fn = @formatSearchDirection;
    end 
    
    if random_attempts > 0
        width = max(nDigitsInWholePart(random_attempts)+1, 2);
    else
        width = 2;
    end
    sd_str_fn = @(code,tries) searchDirectionStr(code,tries,width);
    
    type = struct(                                                      ...
        'label',                'SD',                                   ... 
        'width',                width,                                  ...
        'format_fn',            format_fn,                              ...
        'blank_str',            blanks(width),                          ...
        'na_str',               centerString('-',width)                 );
    
    function str = formatSearchDirection(sd_code,random_attempts)
        str = sprintf('%-*s',width,sd_str_fn(sd_code,random_attempts));
    end
    
    function str = formatSearchDirectionOrange(sd_code,random_attempts)
        str = formatSearchDirection(sd_code,random_attempts);
        
        % print search direction type in orange if it isn't the default
        if sd_code > 2 || (constrained && sd_code > 0)
            str = formatOrange(str);
        end
    end
end

function type = getStationarityMeasure(use_orange)
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