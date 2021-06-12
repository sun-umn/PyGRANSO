function opts = gransoOptions(n,varargin)
%   gransoOptions:
%       Validate user options struct for granso.m.  If user_opts is [] or
%       not provided, returned opts will be GRANSO's default parameters.
%       Standard or advanced options may be set.  
%   
%       Type:
%       >> help gransoOptionsAdvanced 
%       to see documentation for the advanced user options.
%   
%   USAGE:
%       opts = gransoOptions(n);
%       opts = gransoOptions(n,user_opts);
%
%   INPUT:
%       n           Number of variables being optimized.
%
%       user_opts   Struct of settable algorithm parameters.  No fields are 
%                   required, irrelevant fields are ignored, and user_opts 
%                   may be given as [].
%
%   OUTPUT:
%       opts        Struct of all tunable user parameters for GRANSO.
%                   If a field is provided in user_opts, then the user's 
%                   value is checked to whether or not it is a valid value, 
%                   and if so, it is set in opts.  Otherwise, an error is 
%                   thrown.  If a field is not provided in user_opts, opts
%                   will contain the field with GRANSO's default value.  
%
%   STANDARD PARAMETERS 
%
%   .x0                             [n by 1 real vector | {randn(n,1)}]
%       Initial starting point.  One should pick x0 such that the objective
%       and constraint functions are smooth at and about x0.  If this is
%       difficult to ascertain, it is generally recommended to initialize
%       GRANSO at randomly-generated starting points.
%
%   .mu0                            [real > 0 | {1}]
%       Initial value of the penalty parameter.
%       NOTE: irrelevant for unconstrained optimization problems.
%
%   .H0:                            [n by n real matrix | {speye(n)}]
%       Initial inverse Hessian approximation.  In full-memory mode, and 
%       if opts.checkH0 is true, GRANSO will numerically assert that this
%       matrix is positive definite.  In limited-memory mode, that is, if
%       opts.limited_mem_size > 0, no numerical checks are done but this 
%       matrix must be a sparse matrix.
%   
%   .checkH0                        [logical | {true}]
%       By default, GRANSO will check whether or not H0 is numerically
%       positive definite (by checking whether or not chol() succeeds).
%       However, when restarting GRANSO from the last iterate of an earlier
%       run, using soln.H_final (the last BFGS approximation to the inverse
%       Hessian), soln.H_final may sometimes fail this check.  Set this
%       option to false to disable it.  No positive definite check is done
%       when limited-memory mode is enabled.
%
%   .scaleH0                        [logical | {true}]
%       Scale H0 during BFGS/L-BFGS updates.  For full-memory BFGS, scaling
%       is only applied on the first iteration only, and is generally only
%       recommended when H0 is the identity (which is GRANSO's default).
%       For limited-memory BFGS, H0 is scaled on every update.  For more
%       details, see opts.limited_mem_fixed_scaling.
% 
%   .bfgs_damping                   [real in [0,1] | {1e-4}]
%       This feature will adaptively damp potentially problematic BFGS
%       updates in order to help ensure that the (L)BFGS inverse Hessian
%       approximation always remains positive definite numerically.  The
%       closer this value is to one, the more frequently and aggressively
%       damping will be applied to (L)BFGS updates.  Setting this value to
%       zero completely disables damping.
%
%   .limited_mem_size               [nonnegative integer | {0}]
%       By default, GRANSO uses full-memory BFGS updating.  For nonsmooth
%       problems, full-memory BFGS is generally recommended.  However, if
%       this is not feasible, one may optionally enable limited-memory BFGS
%       updating by setting opts.limited_mem_size to a positive integer
%       (significantly) less than the number of variables.
%  
%   .limited_mem_fixed_scaling      [logical | {true}]
%       In contrast to full-memory BFGS updating, limited-memory BFGS
%       permits that H0 can be scaled on every iteration.  By default,
%       GRANSO will reuse the scaling parameter that is calculated on the
%       very first iteration for all subsequent iterations as well.  Set
%       this option to false to force GRANSO to calculate a new scaling
%       parameter on every iteration.  Note that opts.scaleH0 has no effect
%       when opts.limited_mem_fixed_scaling is set to true.
%
%   .limited_mem_warm_start         [struct | {[]}]
%       If one is restarting GRANSO, the previous L-BFGS information can be
%       recycled by setting opts.limited_mem_warm_start = soln.H_final,
%       where soln is GRANSO's output struct from a previous run.  Note
%       that one can either reuse the previous H0 or set a new one.
%       
%   .prescaling_threshold           [real > 0 | {inf}]
%       Pre-scales objective/constraint functions such that the norms of 
%       their gradients evaluated at x0 do not exceed prescaling_threshold.  
%       Inf (default) disables all pre-scaling.  Problems that are poorly
%       scaled, that is, the gradients have norms that are large, may cause 
%       difficulties for optimization.  Pre-scaling can help mitigate these 
%       issues in an automatic way but, ideally, the user should consider 
%       whether an alterative formulation of the optimization problem with 
%       better inherent scaling is possible.  
%       NOTE: solutions obtained for a pre-scaled problem MAY NOT BE a
%       solutions for the originally specified problem.
%
%   .prescaling_info_msg            [logical | {true}]
%       Prints a notice that GRANSO has either automatically pre-scaled at
%       least one of the objective or constraint functions or it has
%       deteced that the optimization problem may be poorly scaled.  For
%       more details, see opts.prescaling_threshold.  
%
%   .opt_tol                        [real >= 0 | {1e-8}]
%       Tolerance for reaching (approximate) optimality/stationarity.
%       See opts.ngrad, opts.evaldist, and the description of GRANSO's 
%       output argument soln, specifically the subsubfield .dnorm for more
%       information.
%
%   .rel_tol                        [real >= 0 | {0}]
%       Tolerance for determining when the relative decrease in the penalty
%       function is sufficiently small.  GRANSO will terminate if when 
%       the relative decrease in the penalty function is at or below this
%       tolerance and the current iterate is feasible to tolerances.
%       Generally, we don't recommend using this feature since small steps
%       are not necessarily indicative of being near a stationary point,
%       particularly for nonsmooth problems.
%
%   .step_tol                       [real > 0 | {1e-12}]
%       Tolerance for how small of a step the line search will attempt
%       before terminating.
%
%   .viol_ineq_tol                  [real >= 0 | {0}]
%       Acceptable total violation tolerance of the inequality constraints.     
%
%   .viol_eq_tol                    [real >= 0 | {1e-6}]
%       Acceptable total violation tolerance of the equality constraints.          
%
%   .ngrad                          [integer > 0 | {min([100, 2*n, n+10])}]
%       Max number of previous gradients to be cached.  The QP defining 
%       GRANSO's measure of stationarity requires a history of previous 
%       gradients.  Note that large values of ngrad can make the related QP
%       expensive to solve, if a significant fraction of the currently
%       cached gradients were evaluated at points within evaldist of the 
%       current iterate.  Using 1 is recommended if and only if the problem 
%       is unconstrained and the objective is known to be smooth.  See 
%       opts.opt_tol, opts.evaldist, and the description of GRANSO's output
%       argument soln, specifically the subsubfield .dnorm for more
%       information.
%
%   .evaldist                       [real > 0 | {1e-4}]
%       Previously evaluated gradients are only used in the stationarity 
%       test if they were evaluated at points that are within distance 
%       evaldist of the current iterate x.  See opts.opt_tol, opts.ngrad, 
%       and the description of GRANSO's output argument soln, specifically 
%       the subsubfield .dnorm for more information.
%
%   .maxit                          [integer > 0 | {1000}]
%       Max number of iterations.
%
%   .maxclocktime                   [real > 0 | {inf}] 
%       Quit if the elapsed clock time in seconds exceeds this.
%                   
%   .fvalquit                       [any real | {-inf}] 
%       Quit if objective function drops below this value at a feasible 
%       iterate (that is, satisfying feasibility tolerances 
%       opts.viol_ineq_tol and opts.viol_eq_tol).
% 
%   .halt_on_quadprog_error         [logical | {false}]
%       By default, GRANSO will attempt to 'work around' any quadprog
%       failure (numerically invalid result or quadprog throws a bonafide
%       error) according to a set of default fallback strategies (see
%       gransoOptionsAdvanced for how these can be configured).  Generally,
%       users should expect quadprog to mostly work, with errors only
%       occurring quite rarely.  However, if quadprog fails frequently,
%       then GRANSO's performance will likely be greatly hindered (in terms
%       of efficiency and quality of optimization).  Set this option to
%       true if one wishes GRANSO to halt on the first quadprog error 
%       encountered while computing the search direction.
%
%   .halt_on_linesearch_bracket     [logical | {true}]
%       If the line search brackets a minimizer but fails to satisfy the 
%       weak Wolfe conditions (necessary for a step to be accepted), GRANSO 
%       will terminate at this iterate when this option is set to true 
%       (default).  For unconstrained nonsmooth problems, it has been 
%       observed that this type of line search failure is often an 
%       indication that a stationarity has in fact been reached.  By 
%       setting this parameter to false, GRANSO will instead first attempt 
%       alternative optimization strategies (if available) to see if
%       further progress can be made before terminating.   See
%       gransoOptionsAdvanced for more details on GRANSO's available 
%       fallback optimization strategies and how they can be configured.
%
%   .quadprog_info_msg              [logical | {true}]
%       Prints a notice that GRANSO's requires a quadprog-compatible QP
%       solver and that the choice of QP solver may affect GRANSO's quality
%       of performance, in terms of efficiency and level of optimization. 
%
%   .print_level                    [integer in {0,1,2,3} | 1]
%       Level of detail printed to console regarding optimization progress:
%           0 - no printing whatsoever
%           1 - prints info for each iteration  
%           2 - adds additional info about BFGS updates and line searches
%           3 - adds info on any errors that are encountered
%
%   .print_frequency                [integer in {1,2,3,...,inf} | 1]
%       Sets how often the iterations are printed.  When set to one, every
%       iteration is printed; when set to 10, only every 10th iteration is
%       printed.  When set to inf, no iterations are printed, except for
%       at x0.  Note that this only affects .print_level == 1 printing;
%       all messages from higher values of .print_level will still be
%       printed no matter what iteration they occurred on.
%
%   .print_width                    [integer in {9,...,23} | {14}]
%       Number of characters wide to print values for the penalty function,
%       the objective function, and the total violations of the inequality 
%       and equality constraints. 
%
%   .print_ascii                    [logical | {false}]
%       By default, GRANSO's printed output uses the extended character
%       map, so nice looking tables can be made.  However, diary() does not
%       capture these symbols.  So, if you need to record the output, you
%       can restrict the printed output to only use the basic ASCII
%       character map, which may look better when captured by diary().
%
%   .print_use_orange               [logical | {true}]
%       By default, GRANSO's printed output makes use of an undocumented
%       MATLAB feature for printing orange text.  GRANSO's uses orange
%       printing to highlight pertinent information.  However, the user
%       is the given option to disable it, since support cannot be
%       guaranteed (since it is an undocumented feature).
%  
%   .halt_log_fn                    [a function handle | {[]}]  
%       A user-provided function handle that is called on every iteration
%       to allow the user to signal to GRANSO for it to halt at that 
%       iteration and/or create historical logs of the progress of the
%       algorithm.  For more details, see also makeHaltLogFunctions in the
%       halt_log_template folder, which shows the function signature
%       halt_log_fn must have if supplied.
%
%   .debug_mode                     [logical | {false}]
%       By default, GRANSO will catch any errors that occur during runtime,
%       in order to be able to return the best computed result so far. 
%       Instead of rethrowing the error, GRANSO will instead print an error
%       message without and add the error object to GRANSO's struct output
%       argument soln.  However, this behavior can make it harder to debug
%       GRANSO so it can be disabled by setting this option to true.
%        
%   END OF STANDARD PARAMETERS
%
%   See also granso, gransoOptionsAdvanced, and makeHaltLogFunctions.
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
%   GRANSO Version 1.6.4, 2016-2020, see AGPL license info below.
%   gransoOptions.m introduced in GRANSO Version 1.0.
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

    persistent default_opts;
    persistent LAST_FALLBACK_LEVEL;
    
    % This will be disabled by the default options or if the user does not
    % activate debug mode
    debug_mode = true;
    
    assert( isAnInteger(n) && n > 0,'GRANSO:invalidUserOption',     ...
            'Number of variables n must be a positive integer!'     );
        
    if isempty(default_opts)
        [default_opts, LAST_FALLBACK_LEVEL] = getDefaults(n);
    end
        
    if nargin < 2 || isempty(varargin{1})
        opts = postProcess(default_opts);
        return
    else
        user_opts = varargin{1};
    end
    
    assert( isstruct(user_opts) && isscalar(user_opts) == 1,        ...
            'GRANSO:invalidUserOption',                             ...
            'GRANSO options must provided as a struct!'             );
    
    % USER PROVIDED THEIR OWN OPTIONS SO WE MUST VALIDATE THEM
    validator = optionValidator('GRANSO',default_opts);
    validator.setUserOpts(user_opts);
    
    % surround the validation so we can rethrow the error from GRANSO
    try
        
        % Set debug mode first as we need its value in the catch block
        debug_mode = validator.getValue('debug_mode');
        validator.setLogical('debug_mode');
        
        % SET INITIAL POINT AND PENALTY PARAMETER VALUE
        if isfield(user_opts,'x0') && ~isempty(user_opts.x0)
            validator.setColumnDimensioned('x0',n);
            validator.setRealFiniteValued('x0');
        end
        validator.setRealNonnegative('mu0');
        
        % SET INITIAL (L)BFGS DATA
        validator.setLogical('checkH0');
        validator.setLogical('scaleH0');
        validator.setRealInIntervalCC('bfgs_damping',0,1);
        validator.setLogical('limited_mem_fixed_scaling');
        validator.setIntegerNonnegative('limited_mem_size');
        lim_mem_size    = validator.getValue('limited_mem_size');
        lim_mem_mode    = lim_mem_size > 0;
        if lim_mem_mode && isfield(user_opts,'limited_mem_warm_start')  ...
                && ~isempty(user_opts.limited_mem_warm_start)
            
            % Ensure all the necessary subfields for L-BFGS data exist and 
            % if so, it returns a validator for this sub-struct of data.
            lbfgs_validator = validator.setStructWithFields(            ...
                'limited_mem_warm_start','S','Y','rho','gamma'          );
            
            ws              = user_opts.limited_mem_warm_start;
            [n_S,cols_S]    = size(ws.S);
            [n_Y,cols_Y]    = size(ws.Y);
            cols_rho        = size(ws.rho,2);
            
            lbfgs_validator.assert(                                     ...
                n == n_S && n == n_Y,                                   ...
                [   'the number of rows in both subfields S and Y must '...
                    'match the number of optimization variables'        ]);
            lbfgs_validator.assert(                                     ...
                cols_S > 0 && cols_S == cols_Y && cols_S == cols_rho,   ...
                [   'subfields S, Y, and rho must all have the same '   ...
                    '(positive) number of columns'                      ]);
            lbfgs_validator.setRow('rho');            
            
            lbfgs_validator.setRealFiniteValued('S');
            lbfgs_validator.setRealFiniteValued('Y');
            lbfgs_validator.setRealFiniteValued('rho');
            lbfgs_validator.setReal('gamma');
            lbfgs_validator.setFiniteValued('gamma');
        end
        if isfield(user_opts,'H0') && ~isempty(user_opts.H0)
            validator.setDimensioned('H0',n,n);
            validator.setRealFiniteValued('H0');
            if lim_mem_mode
                validator.setSparse('H0');
            elseif validator.getValue('checkH0')
                validator.setPositiveDefinite('H0');
            end
        end
        
        % SET PRESCALING PARAMETERS
        validator.setRealPositive('prescaling_threshold');
        validator.setLogical('prescaling_info_msg');

        % CONVERGE CRITERIA / PARAMETERS
        % allow users to set zero optimality and violation tolerances, even 
        % though it's a bit demanding ;-)
        validator.setRealNonnegative('opt_tol');
        validator.setRealNonnegative('rel_tol');
        validator.setRealPositive('step_tol');
        validator.setRealNonnegative('viol_ineq_tol');
        validator.setRealNonnegative('viol_eq_tol');
        validator.setIntegerPositive('ngrad');
        validator.setRealPositive('evaldist');

        % EARLY TERMINATION PARAMETERS
        validator.setIntegerPositive('maxit');
        validator.setRealPositive('maxclocktime');
        validator.setReal('fvalquit');
        validator.setLogical('halt_on_quadprog_error');
        validator.setLogical('halt_on_linesearch_bracket');

        % FALLBACK PARAMETERS (allowable last resort "heuristics")
        validator.setIntegerInRange(    'min_fallback_level',           ...
                                        0,LAST_FALLBACK_LEVEL           );
        % Use the custom validator so we can set a custom message
        validator.validateAndSet(                                       ...
            'max_fallback_level',                                       ...                
            @(x)    isAnInteger(x)     &&                               ...
                    x >= validator.getValue('min_fallback_level') &&    ...
                    x <= LAST_FALLBACK_LEVEL,                           ...
            sprintf('an integer in {opts.min_fallback_level,...,%d}',   ...
                    LAST_FALLBACK_LEVEL)                                );
        validator.setIntegerPositive('max_random_attempts');

        % STEERING PARAMETERS
        validator.setLogical('steering_l1_model');
        validator.setRealNonnegative('steering_ineq_margin');
        validator.setIntegerPositive('steering_maxit');
        validator.setRealInIntervalOO('steering_c_viol',0,1);
        validator.setRealInIntervalOO('steering_c_mu',0,1); 
        validator.setLogical('quadprog_info_msg');
        validator.setStruct('quadprog_opts');
        validator.setRealInIntervalCC('regularize_threshold',1,inf);
        validator.setLogical('regularize_max_eigenvalues');

        % LINE SEARCH PARAMETERS
        % wolfe1: conventionally wolfe1 should be positive in (0,1) but
        % zero is is usually fine in practice (though there are
        % exceptions).  1 is not acceptable.
        % wolfe2: conventionally wolfe2 should be > wolfe1 but it is
        % sometimes okay for both to be zero (e.g. Shor)
        validator.setRealInIntervalCC('wolfe1',0,0.5); 
        validator.setRealInIntervalCO('wolfe2',validator.getValue('wolfe1'),1);                             
        validator.setIntegerNonnegative('linesearch_nondescent_maxit');
        validator.setIntegerNonnegative('linesearch_reattempts');
        validator.setIntegerNonnegative('linesearch_reattempts_x0');
        validator.setRealInIntervalOO('linesearch_c_mu',0,1);    
        validator.setRealInIntervalOO('linesearch_c_mu_x0',0,1);    

        % LOGGING PARAMETERS
        validator.setIntegerInRange('print_level',0,3);
        validator.setIntegerInRange('print_frequency',1,inf);
        validator.setIntegerInRange('print_width',9,23);
        validator.setLogical('print_ascii');
        validator.setLogical('print_use_orange');
                
        if isfield(user_opts,'halt_log_fn') && ~isempty(user_opts.halt_log_fn)
            validator.setFunctionHandle('halt_log_fn');
        end
        
        % Extended ASCII chars in MATLAB on Windows are not monospaced so
        % don't support them.
        opts = validator.getValidatedOpts();
        if ~opts.print_ascii
            validator.assert(~ispc(),                                   ...
                'only opts.print_ascii == true is supported on Windows.');
        end
                
    catch err
        if debug_mode
            err.rethrow();         
        end
        err.throwAsCaller();   
    end
    
    % GET THE VALIDATED OPTIONS AND POST PROCESS THEM
    opts = postProcess(validator.getValidatedOpts());
    
    function opts = postProcess(opts)
        
        % bump up the max fallback level if necessary
        if opts.max_fallback_level < opts.min_fallback_level
            opts.max_fallback_level = opts.max_fallback_level;
        end
        
        % If an initial starting point was not provided, use random vector
        if isempty(opts.x0)
            opts.x0 = randn(n,1);
        end
        % If an initial inverse Hessian was not provided, use the identity
        if isempty(opts.H0)
            opts.H0 = speye(n);
        end
        
        if isfield(opts.quadprog_opts,'QPsolver')
            QPsolver = opts.quadprog_opts.QPsolver;
        end
        
        % MATLAB default solver
        if (strcmp(QPsolver,'quadprog'))
            
            % By default, suppress quadprog's console printing and warnings
            if ~isfield(opts.quadprog_opts,'Display')
                opts.quadprog_opts.Display = 'off';
            end
            
            % Technically the following is a solveQP option, not a quadprog one
            if ~isfield(opts.quadprog_opts,'suppress_warnings')
                opts.quadprog_opts.suppress_warnings = true;
            end
            
        %         Update: By default, suppress qpalm's console printing and warnings
        elseif (strcmp(QPsolver,'qpalm'))
            if ~isfield(opts.quadprog_opts,'print_iter')
                opts.quadprog_opts.verbose = false;
            end
        end
    end

end

function [default_opts, LAST_FALLBACK_LEVEL] = getDefaults(n)
    % get constant for granso.m regarding its last available fallback level
    [~, LAST_FALLBACK_LEVEL]            = gransoConstants();
    
    % default options for GRANSO
    default_opts = struct(                                      ...
        'x0',                           [],                     ...
        'mu0',                          1,                      ...
        'H0',                           [],                     ...
        'checkH0',                      true,                   ...
        'scaleH0',                      true,                   ...
        'bfgs_damping',                 1e-4,                   ...
        'limited_mem_size',             0,                      ...
        'limited_mem_fixed_scaling',    true,                   ...
        'limited_mem_warm_start',       [],                     ...
        'prescaling_threshold',         inf,                    ...
        'prescaling_info_msg',          true,                   ...
        'opt_tol',                      1e-8,                   ...
        'rel_tol',                      0,                      ...
        'step_tol',                     1e-12,                  ...
        'viol_ineq_tol',                1e-6,                   ...
        'viol_eq_tol',                  1e-6,                   ...
        'ngrad',                        min([100, 2*n, n+10]),  ...
        'evaldist',                     1e-4,                   ...
        'maxit',                        1000,                   ...
        'maxclocktime',                 inf,                    ...
        'fvalquit',                     -inf,                   ...
        'halt_on_quadprog_error',       false,                  ...
        'halt_on_linesearch_bracket',   true,                   ...
        'min_fallback_level',           0,                      ...
        'max_fallback_level',           LAST_FALLBACK_LEVEL-1,  ...
        'max_random_attempts',          5,                      ...
        'steering_l1_model',            true,                   ...       
        'steering_ineq_margin',         1e-6,                   ...
        'steering_maxit',               10,                     ...
        'steering_c_viol',              0.1,                    ...
        'steering_c_mu',                0.9,                    ...
        'regularize_threshold',         inf,                    ...
        'regularize_max_eigenvalues',   false,                  ...
        'quadprog_info_msg',            true,                   ...
        'quadprog_opts',                [],                     ...
        'wolfe1',                       1e-4,                   ...
        'wolfe2',                       0.5,                    ...
        'linesearch_nondescent_maxit',  0,                      ...
        'linesearch_reattempts',        0,                      ...
        'linesearch_reattempts_x0',     10,                     ...
        'linesearch_c_mu',              0.5,                    ...
        'linesearch_c_mu_x0',           0.5,                    ...
        'print_level',                  1,                      ...
        'print_frequency',              1,                      ...
        'print_width',                  14,                     ...
        'print_ascii',                  false || ispc(),        ...
        'print_use_orange',             true,                   ...
        'halt_log_fn',                  [],                     ...
        'debug_mode',                   false                   );
end