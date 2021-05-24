function printer = gransoPrinter(opts,n,n_ineq,n_eq)
%   gransoPrinter:
%       Object for handling printing out info for each iteration and
%       messages.
%
%   INPUT:
%       opts                        
%           Struct of necessary parameters from GRANSO that the printer
%           needs to know about, for keeping the iteration printing nicely
%           aligned and what print options are to be used. 
%       .print_ascii
%       .print_use_orange
%       .print_width  
%       .maxit       
%       .limited_mem_size
%       .linesearch_reattempts
%       .max_random_attempts
%       .max_fallback_level
%       .ngrad
%
%       For more information on the above parameters, see gransoOptions and
%       gransoOptionsAdvanced.
% 
%       n 
%           number of optimization variables.
%
%       ineq_constraints    
%           nonnegative number indicating the number of inequality
%           constraints that are present.
%
%       eq_constraints    
%           nonnegative integer indicating the number of equality
%           constraints that are present.
%
%   OUTPUT:
%       An "object", a struct containing the following functions for
%       printing tasks:
%       .msg                
%           Prints a message inside the table; the full width is available
%           and cell arrays can be used to print multiple lines.
%       .msgWidth          
%           Returns the number of chars wide the message area.
%       .init               
%           For printing the initial info at x0.  Requires input arguments: 
%               penfn_at_x      state of the penalty function.  For more
%                               info, see makePenaltyFunction.m
%               stat_value      stationarity measure value
%               n_qps           number of QPs solved
%       .iter           
%           Print the info for the current iterate.  Requires arguments:
%               iter
%               penfn_at_x      state of the penalty function. For more
%                               info, see makePenaltyFunction.m
%               sd_code         integer code indicating which search
%                               direction was used.
%               random_attempts number of random attempts to produce search
%                               direction (this is generally zero)
%               ls_evals        number of functions evaluations incurred in
%                               the line search to find an acceptable
%                               next iterate
%               alpha           step length taken to get this next iterate
%               n_grads         number of gradients used in computing the 
%                               approximate stationarity measure
%               stat_value      value of the (approx) stationarity measure
%               n_qps           number of QP solves incurred to compute 
%                               the (approx) stationarity measure    
%       .summary
%           Prints the objective and total violations (if present) from the
%           supplied data.  Requires a string to be used in the iter column
%           and a struct of the data (provided by provided by the
%           makePenaltyFunction object.
%       .unscaledMsg
%           Prints an informational message about prescaling, since GRANSO
%           prints the prescaled problem values, except at the end for the
%           results, where it prints both the prescaled and unscaled
%           values.
%       .regularizeError
%           Prints a single-line error message to indicate when an eig
%           error causes the regularization procedure to abort.
%       .bfgsError
%           Prints a single-line error message corresponding to the integer
%           code given as the single input argument.  For more information 
%           on these codes, see bfgsHessianInverse.
%       .qpError
%           Prints a multi-line error message showing the details of a
%           thrown error from attempting to solve a QP.  Requires two
%           arguments: the first is the thrown error, the second is a
%           string to be used as a label to indicate to the user which 
%           piece of code threw this error.
%
%
%   If you publish work that uses or refers to GRANSO, please cite the 
%   following paper: 
%
%   [1] Frank E. Curtis, Tim Mitchell, and Michael L. Overton 
%       A BFGS-SQP method for nonsmooth, nonconvex, constrained 
%       optimization and its evaluation using relative minimization 
%       profiles, Optimization Methods and Software, 2016.
%       Available at https://dx.doi.org/10.1080/10556788.2016.1208749
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   gransoPrinter.m introduced in GRANSO Version 1.0.
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

    % Setup printer options from GRANSO options;
    ascii                           = opts.print_ascii;
    use_orange                      = opts.print_use_orange;
    print_opts.use_orange           = use_orange;
    print_opts.print_width          = opts.print_width;
    print_opts.maxit                = opts.maxit;
    print_opts.ls_max_estimate      = 50*(opts.linesearch_reattempts + 1);
    
    [~,LAST_FALLBACK_LEVEL]         = gransoConstants();
    if opts.max_fallback_level < LAST_FALLBACK_LEVEL
        print_opts.random_attempts  = 0;
    else
        print_opts.random_attempts  = opts.max_random_attempts;
    end 
    print_opts.ngrad                = opts.ngrad + 1;
    
    cols        = gransoPrinterColumns(print_opts,n_ineq, n_eq);
    constrained = n_ineq || n_eq;
    if constrained
        pen_label           = 'Penalty Function';
        viol_label          = 'Total Violation';
        pen_vals_fn         = @penaltyFunctionValuesConstrained;
        if n_ineq && n_eq
            viol_vals_fn    = @violationValuesBoth;
        elseif n_ineq
            viol_vals_fn    = @violationValuesInequality;
        else
            viol_vals_fn    = @violationValuesEquality;
        end 
    else
        pen_label           = 'Penalty Fn';
        viol_label          = 'Violation';
        pen_vals_fn         = @penaltyFunctionValues;
        viol_vals_fn        = @violationValues;
    end
   
    iter_c          = cols.iter;
    mu_c            = cols.mu;
    pen_c           = cols.pen;
    obj_c           = cols.obj;
    ineq_c          = cols.ineq;
    eq_c            = cols.eq;
    sd_c            = cols.sd;
    ls_evals_c      = cols.ls_evals;
    ls_size_c       = cols.ls_size;
    ngrad_c         = cols.ngrad;
    stat_c          = cols.stat_value;
            
    labels = {  iter_c.label                                            ...
                mu_c.label      pen_c.label                             ...
                obj_c.label                                             ...
                ineq_c.label    eq_c.label                              ...
                sd_c.label      ls_evals_c.label    ls_size_c.label     ...
                ngrad_c.label   stat_c.label                            };
            
    widths = [  iter_c.width                                            ...
                mu_c.width      pen_c.width                             ...
                obj_c.width                                             ...
                ineq_c.width    eq_c.width                              ...
                sd_c.width      ls_evals_c.width    ls_size_c.width     ...
                ngrad_c.width   stat_c.width                            ];
            
           
    span_labels     = { {pen_label,2,3}                 ...
                        {viol_label,5,6}                ...
                        {'Line Search',7,9}             ...
                        {'Stationarity',10,11}  }; 

    table_printer   = tablePrinter( ascii,      use_orange,             ...
                                    labels,     widths,                 ...
                                    1,          span_labels             );     
    
    print_count     = 0;
            
    msg_box_fn = @(varargin) printMessageBox(ascii,use_orange,varargin{:});
    
    printer = struct(                                               ...                                      
                'msg',                  @table_printer.msg,         ...
                'close',                @table_printer.close,       ...
                'msgWidth',             @table_printer.msgWidth,    ...
                'init',                 @init,                      ...
                'iter',                 @iteration,                 ...
                'summary',              @summary,                   ...
                'unscaledMsg',          @unscaledMsg,               ...
                'lineSearchRestart',    @lineSearchRestart,         ...
                'bfgsInfo',             @printBfgsInfo,             ...
                'regularizeError',      @printRegularizeError,      ...
                'qpError',              @printQPError,              ...
                'quadprogFailureRate',  @quadprogFailureRate        );
         
    function init(penfn_at_x,stat_value,n_qps)
        
        gransoHeader();
        table_printer.header();
        
        print_count = 0;
        pen_values  = pen_vals_fn(penfn_at_x);
        viol_values = viol_vals_fn(penfn_at_x); 
        info_values = infoValues(-1,0,1,0,1);
        
        table_printer.row(  iter_c.format_fn(0),                        ...
                            pen_values{:},                              ...
                            obj_c.format_fn(penfn_at_x.f),              ...
                            viol_values{:},                             ...
                            info_values{:},                             ...
                            stat_c.format_fn(stat_value,n_qps)          );
    end

    function iteration( iter,       penfn_at_x,                         ...
                        sd_code,    random_attempts,                    ...
                        ls_evals,   alpha,                              ...
                        n_grads,    stat_value,     n_qps               )
           
        print_count = print_count + 1;
        if mod(print_count,20) == 0
            table_printer.header();
            print_count = 0;
        end
        
        pen_values  = pen_vals_fn(penfn_at_x);
        viol_values = viol_vals_fn(penfn_at_x); 
        info_values = infoValues(   sd_code,    random_attempts,        ...
                                    ls_evals,   alpha,  n_grads         );
                                
        table_printer.row(  iter_c.format_fn(iter),                     ...
                            pen_values{:},                              ...
                            obj_c.format_fn(penfn_at_x.f),              ...
                            viol_values{:},                             ...
                            info_values{:},                             ...
                            stat_c.format_fn(stat_value,n_qps)          );
    end

    function summary(name,penfn_at_x)   
        viol_values = viol_vals_fn(penfn_at_x); 
        info_values = { sd_c.blank_str,         ls_evals_c.blank_str,   ...
                        ls_size_c.blank_str,    ngrad_c.blank_str       };
                        
        table_printer.row(  iter_c.format_str(name),                    ...
                            mu_c.blank_str,                             ...
                            pen_c.blank_str,                            ...
                            obj_c.format_fn(penfn_at_x.f),              ...
                            viol_values{:},                             ...
                            info_values{:},                             ...
                            stat_c.blank_str                            );
    end

    function unscaledMsg()
        if print_count < 0
            return
        end
        table_printer.overlayOrange(prescalingEndBlockMsg());
    end

    % column entry preprocessing helper functions 

    function pen_args = penaltyFunctionValues(varargin)
        pen_args = {    mu_c.dash_str                       ...
                        pen_c.dash_str                      };
    end

    function pen_args = penaltyFunctionValuesConstrained(penfn_at_x)
        pen_args = {    mu_c.format_fn(penfn_at_x.mu)       ...
                        pen_c.format_fn(penfn_at_x.p)       };
    end

    function viol_args = violationValues(varargin)
        viol_args = {   ineq_c.dash_str                     ...
                        eq_c.dash_str                       };
    end

    function viol_args = violationValuesInequality(penfn_at_x)
        viol_args = {   ineq_c.format_fn(penfn_at_x.tvi)    ...
                        eq_c.dash_str                       };
    end

    function viol_args = violationValuesEquality(penfn_at_x)
        viol_args = {   ineq_c.dash_str                     ...
                        eq_c.format_fn(penfn_at_x.tve)      };
    end

    function viol_args = violationValuesBoth(penfn_at_x)
        viol_args = {   ineq_c.format_fn(penfn_at_x.tvi)    ...
                        eq_c.format_fn(penfn_at_x.tve)      };
    end

    function info_args = infoValues(    sd_code,    random_attempts,    ...
                                        ls_evals,   alpha,              ...
                                        n_grads                         )
                                    
        info_args = {   sd_c.format_fn(sd_code,random_attempts),        ...
                        ls_evals_c.format_fn(ls_evals),                 ...
                        ls_size_c.format_fn(alpha),                     ...
                        ngrad_c.format_fn(n_grads)                      };
    end    

    function lineSearchRestart(iter,mu_ls)
        msg = sprintf([ 'Line search failed to bracket a minimizer, '   ...
                        'reattempting with mu = %g. (Iter = %d)'],      ...
                        mu_ls, iter                                     );
        table_printer.msg(msg); 
    end

    function printBfgsInfo(iter,code)             
        switch code
            case 1
                r = 'applied but without scaling.';
            case 2
                r = 'skipped: sty <= 0.';
            case 3
                r = 'skipped: update contained nans/infs.';
            otherwise
                return
        end
        table_printer.msg(sprintf('BFGS update %s (Iter = %d)',r,iter));
    end

    function printRegularizeError(iter)
        msg = sprintf('Regularization skipped due to eig() error. (Iter = %d)',iter);
        table_printer.msgOrange(msg);
    end

    function printQPError(iter,err,loc)
   
        % title to appear in top/bottom borders
        t1 = sprintf('GRANSO: QUADPROG ERROR IN %s QP (Iter %d) - START',loc,iter);
        t2 = sprintf('GRANSO: QUADPROG ERROR IN %s QP - END',loc);

        err_str                 = getReport(err);
        % Get rid of the stack trace info and just show the root cause
        indx                    = strfind(err_str,'Caused by:');
        if ~isempty(indx)
            err_str             = err_str(indx(1):end);
        end    
        % split the error into a cell of lines
        err_lines               = strsplit(err_str,'\n');
        blank_indx              = cellfun(@isBlankStr,err_lines);
        err_lines(blank_indx)   = [];
        % insert a blank line for the second line
        err_lines               = [{err.message} {''} err_lines];

        fprintf('\n');
        msg_box_fn(1,t1,t2,err_lines,false,80);
        fprintf('\n');
    end

    function quadprogFailureRate(rate)    
        table_printer.msgOrange(quadprogFailureRateMsg(rate));
    end

    % private function to print GRANSO's opening header with name, author,
    % copyright, problem specs, and whether limited-memory mode is active
    function gransoHeader()
        
        table_printer.msg(copyrightNotice());
        
        % print the problem specs
        w       = nDigitsInWholePart(max([n n_ineq n_eq])) + 2;
        spec_fn = @(s,n) sprintf(' %-35s: %*d',sprintf('# of %s',s),w,n);
        
        msg = {                                                         ...
            'Problem specifications:',                                  ...
            spec_fn('variables',n),                                     ...
            spec_fn('inequality constraints',n_ineq),                   ...
            spec_fn('equality constraints',n_eq),                       ...
            };
        table_printer.msg(msg);
        
        % print limited memory message, if enabled
        nvec = opts.limited_mem_size;
        if nvec <= 0
            return
        end
        msg = {                                                         ...
            sprintf('Limited-memory mode enabled with size = %d.',nvec) ...
            [   'NOTE: limited-memory mode is generally NOT '           ...
            'recommended for nonsmooth problems.']                  ...
            };
        table_printer.msgOrange(msg);
    end
end

function s = prescalingEndBlockMsg()
s = {                                                                   ...
''                                                                      ...
'GRANSO applied pre-scaling at x0.  Information:'                       ...
' - ABOVE shows values for the pre-scaled problem'                      ...
' - BELOW shows the unscaled values for the optimization results.'      ...
'NOTE: the pre-scaled solution MAY NOT be a solution to the original'   ...
'unscaled problem!  For more details, see opts.prescaling_threshold.'   ...
''                                                                      ...
};
end

function s = quadprogFailureRateMsg(rate)
s = { ...
'WARNING: GRANSO''s performance may have been hindered by issues with quadprog.'...
sprintf('quadprog''s failure rate: %.3f%%',rate)                                ...
'Ensure that quadprog is working correctly!'                                    ... 
};
end