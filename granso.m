function soln = granso(n,obj_fn,varargin)
%   GRANSO: GRadient-based Algorithm for Non-Smooth Optimization
%
%       Minimize a function, possibly subject to inequality and/or equality 
%       constraints.  GRANSO is intended to be an efficient solver for 
%       constrained nonsmooth optimization problems, without any special 
%       structure or assumptions imposed on the objective or constraint 
%       functions.  It can handle problems involving functions that are any
%       or all of the following: smooth or nonsmooth, convex or nonconvex, 
%       and locally Lipschitz or non-locally Lipschitz.  
%
%       GRANSO only requires that gradients are provided for the objective
%       and constraint functions.  Gradients should always be numerically
%       assessed with a finite difference utility to ensure that they are
%       correct.
%       
%       The inequality constraints must be formulated as 'less than or
%       equal to zero' constraints while the equality constraints must
%       be formulated as 'equal to zero' constraints.  The user is 
%       free to shift/scale these internally in order to specify how 
%       hard/soft each individual constraints are, provided that they
%       respectively remain 'less than or equal' or 'equal' to zero.
%
%       The user must have a quadprog-compatible quadratic program solver,
%       that is, a QP solver that is callable via The MathWorks quadprog 
%       interface (such as quadprog.m from MATLAB's Optimization Toolbox or 
%       MOSEK).  
%
%   NOTE: 
%       On initialization, GRANSO will throw errors if it detects invalid
%       user options or if the user-provided functions to optimize either 
%       do not evaluate or not conform to GRANSO's format.  However, once
%       optimization begins, GRANSO will catch any error that is thrown and
%       terminate normally, so that the results of optimization so far
%       computed can be returned to the user.  This way, the error may be
%       able to corrected by the user and GRANSO can be restarted from the
%       last accepted iterate before the error occurred.
%
%       After GRANSO executes, the user is expected to check all of the
%       following fields:
%           - soln.termination_code
%           - soln.quadprog_failure_rate
%           - soln.error (if it exists)
%       to determine why GRANSO halted and to ensure that GRANSO ran
%       without any issues that may have negatively affected its
%       performance, such as quadprog failing too frequently or a
%       user-provided function throwing an error or returning an invalid
%       result.
%
%   USAGE:
%   - combined_fn evaluates objective and constraints simultaneously:
%
%       % "combined" format
%       soln = GRANSO(n,combined_fn);
%       soln = GRANSO(n,combined_fn,options);
%
%   OR, ALTERNATIVELY:
%   - obj_fn evaluates the objective function while constraint functions 
%     are provided separately via ineq_fn and eq_fn
%       
%       % "separate" format
%       soln = GRANSO(n,obj_fn,ineq_fn,eq_fn);
%       soln = GRANSO(n,obj_fn,ineq_fn,eq_fn,options);
%
%   The first usage may be both more convenient and efficient if computed
%   values appear across the objective and various constraints.    
%
%   INPUT:
%       n               Number of variables to optimize.
%
%       obj_fn          Function handle of single input x, a real-valued
%                       n by 1 vector, for evaluating either:
%
%                       - The values and gradients of the objective and
%                         constraints simultaneously:
%                         [f,f_grad,ci,ci_grad,ce,ce_grad] = obj_fn(x)
%                         In this case, ci and/or ce should be returned as
%                         [] if no (in)equality constraints are given.
% 
%                       - Only the value and gradient of the objective
%                         function to be minimized:
%                         [f,g] = obj_fn(x)
%                         In this case, ineq_fn and eq_fn must be provided.
%                         If there are no (in)equality constraints,
%                         (in)eq_fn should be set as [].
%
%       ineq_fn, eq_fn  Function handles for (in)equality constraints
%                           [ci,ci_grad] = ineq_fn(x)
%                           [ce,ce_grad] = eq_fn(x)
%                       These are required when given obj_fn only evaluates
%                       the objective; they should be respectively set as
%                       [] if no (in)equality constraint is present.
%
%       NOTE: Each function handle returns the value of the function(s) 
%             evaluated at x, as a column vector, along with its 
%             corresponding gradient(s) as a matrix of column vectors.  
%             For example, if there are n variables and p inequality 
%             constraints, then ci must be supplied as a column vector in 
%             R^p while ci_grad must be given as an n by p matrix of p 
%             gradients for the p inequality constraints.
%
%       options         Optional struct of settable parameters or [].
%                       To see available parameters and their descriptions,
%                       type:
%                       >> help gransoOptions
%                       >> help gransoOptionsAdvanced
%
%   OUTPUT:
%       soln            Struct containing the computed solution(s) and
%                       additional information about the computation.
% 
%       If the problem has been pre-scaled, soln will contain:
%
%       .scalings       Struct of pre-scaling multipliers of:
%                         the objective          - soln.scalings.f
%                         inequality constraints - soln.scalings.ci
%                         equality constraints   - soln.scalings.ce
%                       These subsubfields contain real-valued vectors of 
%                       scalars in (0,1] that rescale the corresponding 
%                       function(s) and their gradients.  A multiplier that
%                       is one indicates that its corresponding function
%                       has not been pre-scaled.  The absence of a 
%                       subsubfield {f,ci,ce} indicates that none of the 
%                       functions belonging to that group were pre-scaled.
%
%       The soln struct returns the optimization results in the following
%       fields:
%
%       .final          Function values at the last accepted iterate
%
%       .best           Info for the point that most optimizes the 
%                       objective over the set of iterates encountered
%                       during optimization.  Note that this point may not
%                       be an accepted iterate.  For constrained problems,
%                       this is the point that most optimizes the objective
%                       over the set of points encountered that were deemed 
%                       feasible with respect to the violation tolerances 
%                       (opts.viol_ineq_tol and opts.viol_eq_tol).  If no 
%                       points were deemed feasible to tolerances, then
%                       this field WILL NOT BE present.
%
%       .most_feasible  Info for the best most feasible iterate.  If
%                       iterates are encountered which have zero total
%                       violation, then this holds the best iterate amongst
%                       those, i.e., the one which minimized the objective
%                       function the most.  Otherwise, this contains the 
%                       iterate which is closest to the feasible region, 
%                       that is with the smallest total violation, 
%                       regardless of the objective value.  NOTE: this 
%                       field is only present in constrained problems.
%
%       Note that if the problem was pre-scaled, the above fields will only
%       contain the pre-scaled values.  Furthermore, the solution to the
%       pre-scaled problem may or may not be a solution to the unscaled
%       problem.  For convenience, corresponding unscaled versions of the
%       above results are respectively provided in:
%       .final_unscaled
%       .best_unscaled
%       .most_feasible_unscaled
%
%       For .final, .best, and .most_feasible (un)scaled variants, the 
%       following subsubfields are always present:
%       .x                  the computed iterate
%       .f                  value of the objective at this x
%
%       If the problem is constrained, these subsubfields are also present:
%       .ci                 inequality constraints at x
%       .ce                 equality constraints at x
%       .tvi                total violation of inequality constraints at x
%       .tve                total violation of equality constraints at x
%       .tv                 total violation at x (vi + ve)
%       .feasible_to_tol    true if x is considered a feasible point with
%                           respect to opts.viol_ineq_tol and
%                           opts.viol_eq_tol
%       .mu                 the value of the penalty parameter when this
%                           this iterate was computed
%
%       NOTE: The above total violation measures are with respect to the
%       infinity norm, since the infinity norm is used for determining
%       whether or not a point is feasible.  Note however that GRANSO 
%       optimizes an L1 penalty function.
%
%       The soln struct also has the following subfields:
%       .H_final            Full-memory BFGS:
%                           - The BFGS inverse Hessian approximation at
%                             the final iterate
%                           Limited-memory BFGS:
%                           - A struct containing fields S,Y,rho,gamma, 
%                             representing the final LBFGS state (so that
%                             GRANSO can be warm started using this data).
%       
%       .stat_value         an approximate measure of stationarity at the 
%                           last accepted iterate.  See opts.opt_tol, 
%                           opts.ngrad, opts.evaldist, and equation (13) 
%                           and its surrounding discussion in the paper 
%                           referenced below describing the underlying 
%                           BFGS-SQP method that GRANSO implements.
%   
%       .iters              The number of iterations incurred.  Note that 
%                           if the last iteration fails to produce a step,
%                           it will not be printed.  Thus, this value may 
%                           sometimes appear to be greater by 1 compared to 
%                           the last iterate printed.
%
%       .BFGS_updates       A struct of data about the number of BFGS 
%                           updates that were requested and accepted.
%                           Numerical issues may force updates to be
%                           (rarely) skipped.
%       
%       .fn_evals           The total number of function evaluations
%                           incurred.  Evaluating the objective and
%                           constraint functions at a single point counts
%                           as one function evaluation.
%   
%       .termination_code   Numeric code indicating why GRANSO terminated:
%           0:  Approximate stationarity measurement <= opts.opt_tol and 
%               current iterate is sufficiently close to the feasible 
%               region (as determined by opts.viol_ineq_tol and 
%               opts.viol_eq_tol).
%
%           1:  Relative decrease in penalty function <= opts.rel_tol and
%               current iterate is sufficiently close to the feasible 
%               region (as determined by opts.viol_ineq_tol and 
%               opts.viol_eq_tol).
%
%           2:  Objective target value reached at an iterate
%               sufficiently close to feasible region (determined by
%               opts.fvalquit, opts.viol_ineq_tol and opts.viol_eq_tol).
%
%           3:  User requested termination via opts.halt_log_fn 
%               returning true at this iterate.
%
%           4:  Max number of iterations reached (opts.maxit).
%
%           5:  Clock/wall time limit exceeded (opts.maxclocktime).
%
%           6:  Line search bracketed a minimizer but failed to satisfy
%               Wolfe conditions at a feasible point (with respect to
%               opts.viol_ineq_tol and opts.viol_eq_tol).  For 
%               unconstrained problems, this is often an indication that a
%               stationary point has been reached.
%
%           7:  Line search bracketed a minimizer but failed to satisfy
%               Wolfe conditions at an infeasible point.
%
%           8:  Line search failed to bracket a minimizer indicating the
%               objective function may be unbounded below.  For constrained
%               problems, where the objective may only be unbounded off the
%               feasible set, consider restarting GRANSO with opts.mu0 set
%               lower than soln.mu_lowest (see its description below for 
%               more details).
%
%           9:  GRANSO failed to produce a descent direction.
%
%           10: NO LONGER IN USE (Previously, it was used to indicate if
%               any of the user-supplied functions returned inf/NaN at x0.
%               Now, GRANSO throws an error back to the user if this
%               occurs).
%
%           11: User-supplied functions threw an error which halted GRANSO.
%
%           12: A quadprog error forced the steering procedure to be 
%               aborted and GRANSO was halted (either because there were no 
%               available fallbacks or opts.halt_on_quadprog_error was set 
%               to true).  Only relevant for constrained problems.
%
%           13: An unknown error occurred, forcing GRANSO to stop.  Please 
%               report these errors to the developer.
%
%   .quadprog_failure_rate  Percent of the time quadprog threw an error or
%                           returned an invalid result.  From 0 to 100.
%
%   .error                  Only present for soln.termination equal to 11, 
%                           12, or 13.  Contains the thrown error that
%                           caused GRANSO to halt optimization.
%
%   .mu_lowest              Only present for constrained problems where
%                           the line search failed to bracket a minimizer 
%                           (soln.termination_code == 7).  This contains
%                           the lowest value of mu tried in the line search
%                           reattempts.  For more details, see the line
%                           search user options by typing:
%                           >> help gransoOptionsAdvanced
%
%   CONSOLE OUTPUT:
%       When opts.print_level is at least 1, GRANSO will print out
%       the following information for each iteration:
%   
%       Iter            The current iteration number
% 
%       Penalty Function (only applicable if the problem is constrained)
%           Mu          The current value of the penalty parameter 
%           Value       Current value of the penalty function
%           
%           where the penalty function is defined as
%               Mu*objective + total_violation
%
%       Objective       Current value of the objective function
%       
%       Total Violation (only applicable if the problem is constrained)
%           Ineq        Total violation of the inequality constraints
%           Eq          Total violation of the equality constraints
%       
%           Both are l-infinity measures that are used to determine
%           feasibility.
%
%       Line Search
%           SD          Search direction type:
%                           S   Steering with BFGS inverse Hessian
%                           SI  Steering with identity 
%                           QN  Standard BFGS direction 
%                           GD  Gradient descent
%                           RX  Random where X is the number of random 
%                               search directions attempted 
%           
%                       Note that S and SI are only applicable for
%                       constrained problems.
%
%                       If the accepted search direction was obtained via a
%                       fallback strategy instead of the standard strategy,
%                       the search direction type will be printed in
%                       orange.  The standard search direction types for
%                       constrained and unconstrained problems respectively
%                       is S and QN.  The standard search direction can be
%                       modified via opts.min_fallback_level.  
% 
%                       The frequent use of fallbacks may indicate a
%                       deficient or broken quadprog installation (or that 
%                       the license is invalid or can't be verified).
%           
%           Evals       Number of points evaluated in line search
%   
%           t           Accepted length of line search step 
%
%       Stationarity    Approximate stationarity measure
%           Grads       Number of gradients used to compute measure
%       
%           Value       Value of the approximate stationarity measure.  
%               
%                       If fallbacks were employed to compute the
%                       stationarity value, that is quadprog errors were 
%                       encountered, its value will be printed in 
%                       orange, with ':X' appearing after it.  The X 
%                       indicates the number of requests to quadprog.  If
%                       the value could not be computed at all, it will be
%                       reported as Inf.
%
%                       The frequent use of fallbacks may indicate a
%                       deficient or broken quadprog installation (or that 
%                       the license is invalid or can't be verified).
%
%   See also gransoOptions, gransoOptionsAdvanced, makeHaltLogFunctions.
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
%   NOTE: GRANSO in all capitals refers to the software package and is the
%   form that should generally be used.  granso or granso.m in lowercase
%   letters refers specifically to the GRANSO routine/command.
%
%   GRANSO uses modifed versions of the BFGS inverse Hessian approximation
%   update formulas and the inexact weak Wolfe line search from HANSO v2.1.
%   See the documentation of HANSO for more information on the use of
%   quasi-Newton methods for nonsmooth unconstrained optimization.
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   GRANSO Version 1.6.4, 2016-2020, see AGPL license info below.
%   granso.m introduced in GRANSO Version 1.0.
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

    if nargin < 2
        error(  'GRANSO:inputArgumentsMissing',     ...
                'not all input arguments provided.' );
    end
    
    % First reset solveQP's persistent counters to zero
    clear solveQP;
    
    % Initialization
    % - process arguments
    % - set initial Hessian inverse approximation
    % - evaluate functions at x0
    try 
        [problem_fns,opts] = processArguments(n,obj_fn,varargin{:});
        [bfgs_hess_inv_obj,opts] = getBfgsManager(opts);
      
        % construct the penalty function object and evaluate at x0
        % unconstrained problems will reset mu to one and mu will be fixed
        [ penaltyfn_obj, ...
          grad_norms_at_x0] = makePenaltyFunction(opts, problem_fns{:});
    catch err
        switch err.identifier
            case 'GRANSO:invalidUserOption'
                printRed('GRANSO: invalid user option.\n');
                err.throwAsCaller();
            case 'GRANSO:userSuppliedFunctionsError'
                displayError(false,userSuppliedFunctionsErrorMsg(),err);
                err.cause{1}.rethrow();
            otherwise
                printRed('GRANSO: ');
                printRed(unknownErrorMsg());
                fprintf('\n');
                err.rethrow();
        end
    end
    
    msg_box_fn = @(varargin) printMessageBox(   opts.print_ascii,       ...
                                                opts.print_use_orange,  ...
                                                varargin{:}             );
    
    print_notice_fn = @(title,msg) msg_box_fn(2,title,[],msg,true);  
    if opts.print_level
        fprintf('\n');
        if opts.quadprog_info_msg
            print_notice_fn('QUADPROG NOTICE',quadprogInfoMsg());
        end
        if opts.prescaling_info_msg
            printPrescalingMsg( opts.prescaling_threshold,      ...
                                grad_norms_at_x0,               ...
                                print_notice_fn                 );
        end
    end
    
    printer = [];
    if opts.print_level 
        n_ineq          = penaltyfn_obj.numberOfInequalities();
        n_eq            = penaltyfn_obj.numberOfEqualities();
        constrained     = n_ineq || n_eq;
        printer         = gransoPrinter(opts,n,n_ineq,n_eq);
    end
       
    try
        info = bfgssqp(penaltyfn_obj,bfgs_hess_inv_obj,opts,printer);
    catch err
        if opts.debug_mode
            err.rethrow();
        end
        info.error  = err;  
        switch err.identifier
            case 'GRANSO:userSuppliedFunctionsError' 
                info.termination_code = 11;
            case 'GRANSO:steeringQuadprogFailure'
                info.termination_code = 12;
            otherwise % unknown error
                info.termination_code = 13;
        end    
        % recover optimization computed so far
        penaltyfn_obj.restoreSnapShot();
    end
    
    % package up solution in output argument
    [ soln, stat_value ]        = penaltyfn_obj.getBestSolutions();
    soln.H_final                = bfgs_hess_inv_obj.getState();
    soln.stat_value             = stat_value;
    bfgs_counts                 = bfgs_hess_inv_obj.getCounts();
    soln.iters                  = bfgs_counts.requests;
    soln.BFGS_updates           = bfgs_counts;
    soln.fn_evals               = penaltyfn_obj.getNumberOfEvaluations();
    soln.termination_code       = info.termination_code;
    [qp_requests,qp_errs]       = solveQP('counts');
    qp_fail_rate                = 100 * (qp_errs / qp_requests);
    soln.quadprog_failure_rate  = qp_fail_rate;
    if isfield(info,'error')
        soln.error              = info.error;
    elseif isfield(info,'mu_lowest')
        soln.mu_lowest          = info.mu_lowest;
    end
          
    if opts.print_level         
        printer.msg({ 'Optimization results:'; getResultsLegend() });
        
        printSummary('F','final');
        printSummary('B','best');
        printSummary('MF','most_feasible');
        if penaltyfn_obj.isPrescaled()
            printer.unscaledMsg();
            printSummary('F','final_unscaled');
            printSummary('B','best_unscaled');
            printSummary('MF','most_feasible_unscaled');
        end
        width = printer.msgWidth();
        printer.msg(getTerminationMsgLines(soln,constrained,width));

        if qp_fail_rate > 1
            printer.quadprogFailureRate(qp_fail_rate);
        end
        printer.close(); 
    end
         
    if isfield(soln,'error')
        err = soln.error;      
        partial_computation = soln.iters > 0;
        switch err.identifier
            case 'GRANSO:userSuppliedFunctionsError' 
                get_msg_fn = @userSuppliedFunctionsErrorMsg;
            case 'GRANSO:steeringQuadprogFailure'
                get_msg_fn = @quadprogErrorMsg;
            otherwise
                get_msg_fn = @unknownErrorMsg;
        end
        displayError(partial_computation,get_msg_fn(),err);
        if ~partial_computation
            if ~isempty(err.cause)
                err.cause{1}.rethrow();
            else
                err.rethrow();
            end
        end
    end 
    
    function printSummary(name,fieldname)
        if isfield(soln,fieldname)
            printer.summary(name,soln.(fieldname));
        end
    end
end

function [problem_fns,options] = processArguments(n,obj_fn,varargin)
    if nargin > 3
        [ineq_fn,eq_fn] = deal(varargin{1:2});
        problem_fns     = {obj_fn,ineq_fn,eq_fn};
        options_index   = 3;  
    else
        problem_fns     = {obj_fn};
        options_index   = 1;
    end
    if length(varargin) < options_index
        options         = [];
    else
        options         = varargin{options_index};
    end
    options             = gransoOptions(n,options);
end

function [bfgs_obj,opts] = getBfgsManager(opts)
    
    if opts.limited_mem_size == 0
        get_bfgs_fn = @bfgsHessianInverse;
        lbfgs_args  = {};
    else
        get_bfgs_fn = @bfgsHessianInverseLimitedMem;
        lbfgs_args  = {     opts.limited_mem_fixed_scaling,     ...
                            opts.limited_mem_size,              ...
                            opts.limited_mem_warm_start         };
    end
    bfgs_obj        = get_bfgs_fn(opts.H0,opts.scaleH0,lbfgs_args{:});
    
    % remove potentially large and unnecessary data from the opts structure
    opts            = rmfield(opts,'H0');
    opts            = rmfield(opts,'limited_mem_warm_start');
end

function printPrescalingMsg(prescaling_threshold,grad_norms,block_msg_fn)
    threshold       = 100;
    prescaling_set  = prescaling_threshold < inf;
    if prescaling_set 
        threshold   = prescaling_threshold;
    end
    
    f_large     = grad_norms.f > threshold;
    ci_large    = grad_norms.ci > threshold;
    ce_large    = grad_norms.ce > threshold;
    large_norms =  f_large || any(ce_large) || any(ce_large);
    
    if large_norms
        if prescaling_set 
            [title,pre_lines,post_lines] = prescalingEnabledMsgs();
        else
            [title,pre_lines,post_lines] = poorScalingDetectedMsgs();
        end
    else
        return
    end

    objective   = ternOp(f_large, {' - the objective'}, {});
   
    max_n       = max([find(ci_large,1,'last') find(ce_large,1,'last')]);
    if isempty(max_n)
        max_n   = 0;
    end
    width       = nDigitsInWholePart(max_n) + 1;
    cols        = max(floor(50 / width),1);
    
    msg         = [                                                     ...
        pre_lines                                                       ...
        {''}                                                            ...
        objective                                                       ...
        getPrescalingConstraintLines('inequality',ci_large,width,cols)  ...
        getPrescalingConstraintLines('equality',ce_large,width,cols)    ...
        {''}                                                            ...
        post_lines                                                      ];
    
    block_msg_fn(title,msg);
    fprintf('\n');
end

function lines = getPrescalingConstraintLines(type_str,c_large,width,cols)
    count   = sum(c_large);
    if count == 0
        lines = {};
        return
    end
    n       = length(c_large);
    title   = sprintf(' - %s constraints (%d of %d):',type_str,count,n);
    table   = getTableRows(find(c_large),width,cols,3,true);
    lines   = [ {title} table ];
end

function rows = getTableRows(nums,num_width,cols,indent,brackets)  
    if ~isempty(nums)
        n_rows  = ceil(length(nums) / cols);
        rows    = cell(1,n_rows);
        line    = cell(1,cols);
        
        spacer  = repmat(' ',1,indent+brackets);
        prefix  = ternOp(brackets, sprintf('%s[',spacer(1:end-1)), spacer);
   
        for j = 1:length(nums)
            col             = mod(j-1,cols) + 1;
            row             = ceil(j / cols);
            line{col}       = sprintf('%*d',num_width,nums(j));
            if j == length(nums) 
                postfix     = ternOp(brackets, ' ]', '');
                rows{row}   = [prefix line{:} postfix];
            elseif col == cols
                rows{row}   = [prefix line{:}];
                line        = cell(1,cols);
                prefix      = spacer;
            end
        end
    else
        rows = {};
    end
end

function msg = quadprogInfoMsg()
msg = {                                                                                 ...
'GRANSO requires a quadratic program (QP) solver that has a quadprog-compatible '       ...
'interface, as defined by MATLAB''s own QP solver quadprog (available in the '          ...
'Optimization Toolbox).  MOSEK also provides a quadprog-compatible interface to their ' ...
'QP solver and, of course, one may always write a quadprog-compatible wrapper for the ' ...
'QP solver of one''s choice.'                                                           ...
'',                                                                                     ...
'Note that GRANSO''s performance, in terms of efficiency and/or optimization quality, ' ...
'may vary depending upon which QP solver is employed, particularly for nonsmooth'       ...
'constrained optimization problems.'                                                    ...
''                                                                                      ...
'WARNING: While GRANSO will attempt fallback strategies to handle intermittent '        ...
'failures (such as a quadprog error), it is nonetheless critical for performance that ' ...
'the available quadprog solver is working correctly.  GRANSO returns the failure rate ' ...
'of quadprog in soln.quadprog_failure_rate.  If the failure rate exceeds 1%, GRANSO '   ...
'will also print out a warning message after optimization has terminated.  The use of ' ...
'fallback strategies on any given iteration is often an indication that one or more '   ...
'quadprog errors have occurred.  See the ''CONSOLE OUTPUT'' section of ''help granso'' '...
'for more details on how to interpret GRANSO''s printed output indicating when and '    ...
'where fallbacks have occurred.'                                                        ...
''                                                                                      ...
'To disable this notice, set opts.quadprog_info_msg = false.'                           ...
};
end

function [title,pre,post] = poorScalingDetectedMsgs()
title = 'POOR SCALING DETECTED';
pre = {                                                                                 ...
'The supplied problem appears to be poorly scaled at x0, which may adversely affect'    ...
'optimization quality.  In particular, the following functions have gradients whose'    ...
'norms evaluted at x0 are greater than 100:'                                            ...
};
post = {                                                                                ...
'NOTE: One may wish to consider whether the problem can be alternatively formulated'    ...
'with better inherent scaling, which may yield improved optimization results.'          ...
'Alternatively, GRANSO can optionally apply automatic pre-scaling to poorly-scaled'     ...
'objective and/or constraint functions if opts.prescaling_threshold is set to some'     ...
'sufficiently small positive number (e.g. 100).  For more details, see gransoOptions.'  ...   
''                                                                                      ...
'To disable this notice, set opts.prescaling_info_msg = false.'                         ...
};
end

function [title,pre,post] = prescalingEnabledMsgs()
title = 'PRE-SCALING ENABLED';      
pre = {                                                                                 ...                                                                                ...
'GRANSO has applied pre-scaling to functions whose norms were considered large at x0.'  ...
'GRANSO will now try to solve this pre-scaled version instead of the original problem'  ...
'given.  Specifically, the following functions have been automatically scaled'          ...
'downward so that the norms of their respective gradients evaluated at x0 no longer'    ...
'exceed opts.prescaling_threshold and instead are now equal to it:'                     ... 
};
post = {                                                                                ...
'NOTE: While automatic pre-scaling may help ameliorate issues stemming from when the'   ...
'objective/constraint functions are poorly scaled, a solution to the pre-scaled'        ...
'problem MAY OR MAY NOT BE A SOLUTION to the original unscaled problem.  One may wish'  ...
'to consider if the problem can be reformulated with better inherent scaling.  The'     ...
'amount of pre-scaling applied by GRANSO can be tuned, or disabled completely, via'     ...
'adjusting opts.prescaling_threshold.  For more details, see gransoOptions.'            ...    
''                                                                                      ...
'To disable this notice, set opts.prescaling_info_msg = false.'                         ...
};
end

function lines = getTerminationMsgLines(soln,constrained,width)
    switch soln.termination_code
        case 0
            s = convergedToTolerancesMsg(constrained);
        case 1
            s = progressSlowMsg(constrained);
        case 2
            s = targetValueAttainedMsg(constrained);
        case 3
            s = 'halt signaled by user via opts.halt_log_fn.';
        case 4
            s = 'max iterations reached.';
        case 5
            s = 'clock/wall time limit reached.';
        case 6
            s = bracketedMinimizerFeasibleMsg(soln,constrained);          
        case 7
            s = bracketedMinimizerInfeasibleMsg();
        case 8
            s = failedToBracketedMinimizerMsg(soln);
        case 9
            s = 'failed to produce a descent direction.';  
        % Case 10 is no longer used
        case 11
            s = 'user-supplied functions threw an error.';
        case 12
            s = 'steering aborted due to quadprog() error; see opts.halt_on_quadprog_error.';
        otherwise
            s = 'unknown termination condition.';
    end
    s = sprintf('GRANSO termination code: %d --- %s',soln.termination_code,s);    
    lines = [   sprintf('Iterations:              %d',soln.iters);      ...
                sprintf('Function evaluations:    %d',soln.fn_evals);   ...
                wrapToLines(s,width,0)                                  ]; 
end

function s = getResultsLegend()
    s = 'F = final iterate, B = Best (to tolerance), MF = Most Feasible';
end

function s = convergedToTolerancesMsg(constrained)
    if constrained
        s = 'converged to stationarity and feasibility tolerances.';
    else
        s = 'converged to stationarity tolerance.';
    end
end

function s = progressSlowMsg(constrained)
    if constrained
        s = 'relative decrease in penalty function is below tolerance and feasibility tolerances satisfied.';
    else
        s = 'relative decrease in objective function is below tolerance.';
    end
end

function s = targetValueAttainedMsg(constrained)
    if constrained
        s = 'target objective reached at point feasible to tolerances.';
    else
        s = 'target objective reached.';
    end
end
 
function s = bracketedMinimizerFeasibleMsg(soln,constrained)
    if constrained
        if soln.final.tv == 0
        s2 = ' at a (strictly) feasible point. ';    
        else
        s2 = ' at a feasible point (to tolerances). ';
        end
    else
        s2 = '. ';
    end
    s = [   ...
    'line search bracketed a minimizer but failed to satisfy Wolfe conditions'  ...
    s2                                                                          ...
    'This may be an indication that approximate stationarity has been attained.'...
    ];
end

function s = bracketedMinimizerInfeasibleMsg()
    s = [   ...
    'line search bracketed a minimizer but failed to satisfy Wolfe conditions ' ...
    'at an infeasible point. The closest point encountered to the feasible '    ...
    'region is available in soln.most_feasible.'                                ...
    ];
end

function s = failedToBracketedMinimizerMsg(soln)
    if isfield(soln,'mu_lowest')
        s_mu = sprintf([   ...
        'GRANSO attempted mu values down to %g unsuccessively.  However, if '   ...
        'the objective function is indeed bounded below on the feasible set, '  ...
        'consider restarting GRANSO with opts.mu0 set even lower than %g.'],    ...
        soln.mu_lowest, soln.mu_lowest);
    else
        s_mu = '';
    end

    s = [   ...
    'line search failed to bracket a minimizer, indicating that the objective ' ...
    'function may be unbounded below. '  s_mu                                   ...
    ];
end

function displayError(partial_computation,error_msg,err)

    if partial_computation
        computation_msg = partialComputationMsg();
        full_report     = true;
    else
        computation_msg = noComputationMsg();
        full_report     = false;
    end
    printRed(computation_msg);
    fprintf('\n');
    
    printRed('HIGH LEVEL CAUSE OF ERROR\n');
    stack_str = displayErrorRecursive(err,full_report);
    printRed('\nSUGGESTED COURSE OF ACTION\n');
    printRed(error_msg);
    fprintf('\n\n');
    printRed('See stack trace below for the exact origin and specifics of error.\n\n%s',stack_str);
end

function s = displayErrorRecursive(err,full_report)
    if ~isempty(err.cause)
        s = displayErrorRecursive(err.cause{1},full_report);
    elseif full_report
        s = getReport(err);     
    else
        s = '';
    end
    identity = err.identifier;
    if strncmp('GRANSO',identity,6)
        identity = 'GRANSO';
    end
    printRed('%s - %s\n',identity,err.message);
end

function s = partialComputationMsg()
s = [   ...
'GRANSO: optimization halted on error.\n'                                                   ...
'  - Output argument soln.final contains last accepted iterate prior to error\n'            ...
'  - See soln.error and console output below for more details on the cause of the error\n'  ...  
'  - GRANSO may be restarted at the last iterate by setting:\n\n'                           ...
'        opts.x0  = soln.final.x\n'                                                         ...
'        opts.mu0 = soln.final.mu\n'                                                        ...
'        opts.H0  = soln.H_final\n\n'                                                       ...
'    See opts.limited_mem_warm_start for restarting when limited-memory mode is in use.\n'  ...
'    NOTE: If opts.prescalingthreshold < inf, GRANSO may choose different pre-scalings \n'  ...
'    when restarting from different starting points.  To ensure scalings are preserved \n'  ...
'    for restarted runs, you must: \n\n'                                                    ...
'        1) disable the pre-scaling option by setting opts.prescaling_threshold = inf. \n'  ...
'        2) if pre-scaling was applied on the previous run, you must also manually \n'      ...
'           provide the pre-scaled versions of your objective and constraint functions, \n' ...
'           using the scaling multipliers stored in soln.scalings \n'                       ...     
];
end

function s = noComputationMsg()
s = [   ...
'GRANSO: error on initialization.\n'                                    ...
'  - See console output below for more details on the cause of error.\n'...
];
end

function s = userSuppliedFunctionsErrorMsg()
s = [   ...
'Please check your supplied routines defining the objective and constraints \n' ...
'functions for correctness.\n'                                                  ...
];
end

function s = quadprogErrorMsg()
s = [   ...
'Incurring a quadprog error may be an indication of:\n'                                     ...
'  a) a weakness/flaw in the specific QP solver being used\n'                               ...
'  b) numerical loss of accuracy in GRANSO, which in turn causes the QP solver to fail\n'   ...
'  c) a bug.\n\n'                                                                           ...
'If you suspect you have found a bug, please report on the GRANSO GitLab webpage:\n'        ...
'    https://gitlab.com/timmitchell/GRANSO\n\n'                                             ...
'However, please first rule out that neither your underlying QP solver nor expected \n'     ...
'numerical issues are to blame.  For the latter, the BFGS approximations to the inverse \n' ...
'Hessian are expected to become extremely ill-conditioned on nonsmooth problems and may \n' ...
'even lose positive definitness numerically, both of which can cause difficulties for QP \n'...
'solvers.  GRANSO can limit the condition numbers of the BFGS inverse Hessian \n'           ...
'approximations to always be at most some finite positve value k and additionally ensure \n'...
'that they remain numerically positive definite. This feature may be enabled by setting \n' ...
'GRANSO options: \n\n'                                                                      ...
'  opts.regularize_threshold     = k      %% finite strictly positive value\n'              ...
'  opts.regularize_max_threshold = false \n\n'                                              ...
'Furthermore, instead of halting on quadprog errors, GRANSO can also be set to \n'          ...
'automatically first try alternative optimization strategies before terminating, which \n'  ...
'may allow GRANSO to "push beyond" such errors.  This can be particularly beneficial if \n' ...
'such errors are sporadic.  This feature may be enabled by setting GRANSO options: \n\n'    ...
'  opts.halt_on_quadprog_error   = false\n'                                                 ...
'  opts.max_fallback_level       = a value in {1,2,3,4} % for constrained problems\n'       ...
'  opts.max_fallback_level       = a value in {3,4}     % for unconstrained problems\n'     ...
];  
end

function s = unknownErrorMsg()
s = [                                                               ...
    'An unknown error has incurred.  '                              ...
    'Please report it on GRANSO''s GitLab page.\n'                  ...  
    '    https://gitlab.com/timmitchell/GRANSO\n\n'                 ...
];
end