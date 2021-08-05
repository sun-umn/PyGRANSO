function info = bfgssqp(    penaltyfn_obj,                  ...
                            bfgs_obj,                       ...
                            opts,                           ...
                            printer                         )
%   bfgssqp:
%       Minimizes a penalty function.  Note that bfgssqp operates on the
%       objects it takes as input arguments and bfgssqp will modify their
%       states.  The result of bfgssqp's optimization process is obtained
%       by querying these objects after bfgssqp has been run.
%
%   INPUT:
%       penaltyfn_obj       
%           Penalty function object from makePenaltyFunction.m
%
%       bfgs_obj
%           (L)BFGS object from bfgsHessianInverse.m or
%           bfgsHessianInvereLimitedMem.m
%
%       opts
%           A struct of parameters for the software.  See gransoOptions.m
% 
%       printer
%           A printer object from gransoPrinter.m
%   
%   OUTPUT:
%       info    numeric code indicating termination circumstance:
%
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
%   bfgssqp.m introduced in GRANSO Version 1.0.
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

    % "Constants" for controlling fallback levels
    % currently these are 2 and 4 respectively
    [POSTQP_FALLBACK_LEVEL, LAST_FALLBACK_LEVEL] = gransoConstants();
                   
    % initialization parameters
    x                           = opts.x0;
    n                           = length(x);
    full_memory                 = opts.limited_mem_size == 0;
    damping                     = opts.bfgs_damping;
    
    % convergence criteria termination parameters
    % violation tolerances are checked and handled by penaltyfn_obj
    opt_tol                     = opts.opt_tol;
    rel_tol                     = opts.rel_tol;
    step_tol                    = opts.step_tol;
    ngrad                       = opts.ngrad;
    evaldist                    = opts.evaldist;

    % early termination parameters
    maxit                       = opts.maxit;
    maxclocktime                = opts.maxclocktime;
    if maxclocktime < inf       
        t_start = tic(); 
    end
    fvalquit                    = opts.fvalquit;
    halt_on_quadprog_error      = opts.halt_on_quadprog_error;
    halt_on_linesearch_bracket  = opts.halt_on_linesearch_bracket;

    % fallback parameters - allowable last resort "heuristics"
    min_fallback_level          = opts.min_fallback_level;
    max_fallback_level          = opts.max_fallback_level;
    max_random_attempts         = opts.max_random_attempts;

    % steering parameters
    steering_l1_model           = opts.steering_l1_model;
    steering_ineq_margin        = opts.steering_ineq_margin;
    steering_maxit              = opts.steering_maxit;
    steering_c_viol             = opts.steering_c_viol;
    steering_c_mu               = opts.steering_c_mu;
    
    % parameters for optionally regularizing of H 
    regularize_threshold        = opts.regularize_threshold;
    regularize_max_eigenvalues  = opts.regularize_max_eigenvalues;
    
    quadprog_opts               = opts.quadprog_opts;

    % line search parameters
    wolfe1                      = opts.wolfe1;
    wolfe2                      = opts.wolfe2;
    linesearch_nondescent_maxit = opts.linesearch_nondescent_maxit;
    linesearch_reattempts       = opts.linesearch_reattempts;
    linesearch_reattempts_x0    = opts.linesearch_reattempts_x0;
    linesearch_c_mu             = opts.linesearch_c_mu;
    linesearch_c_mu_x0          = opts.linesearch_c_mu_x0;

    % logging parameters
    print_level                 = opts.print_level;
    print_frequency             = opts.print_frequency;
    halt_log_fn                 = opts.halt_log_fn;
    user_halt                   = false;
  
    % get value of penalty function at initial point x0 
    % mu will be fixed to one if there are no constraints.
    iter            = 0;
    [f,g]           = penaltyfn_obj.getPenaltyFunctionValue();
    mu              = penaltyfn_obj.getPenaltyParameter();
    constrained     = penaltyfn_obj.hasConstraints();
    
    % The following will save all the function values and gradients for
    % the objective, constraints, violations, and penalty function
    % evaluated at the current x and value of the penalty parameter mu.
    % If a search direction fails, the code will "roll back" to this data
    % (rather than recomputing it) in order to attempt one of the fallback
    % procedures for hopefully obtaining a usable search direction.
    % NOTE: bumpFallbackLevel() will restore the last executed snapshot. It
    % does NOT need to be provided.
    penaltyfn_at_x  = penaltyfn_obj.snapShot();
                                        
    % regularizes H for QP solvers but only if cond(H) > regularize_limit
    % if isinf(regularize_limit), no work is done
    if full_memory && regularize_threshold < inf
        get_apply_H_QP_fn   = @getApplyHRegularized;
    else
        % No regularization option for limited memory BFGS 
        get_apply_H_QP_fn   = @getApplyH;
    end
    [apply_H_QP_fn, H_QP]   = get_apply_H_QP_fn();
    % For applying the normal non-regularized version of H
    [apply_H_fn]            = getApplyH();
    
    bfgs_update_fn          = @bfgs_obj.update;

    % function which caches up to ngrad previous gradients and will return 
    % those which are sufficently close to the current iterate x. 
    % The gradients from the current iterate are simultaneously added to 
    % the cache. 
    get_nbd_grads_fn        = neighborhoodCache(ngrad,evaldist);
    get_nearby_grads_fn     = @() getNearbyGradients(   penaltyfn_obj,  ...
                                                        get_nbd_grads_fn);
    [stat_vec,stat_val,qps_solved] = computeApproxStationarityVector();
   
    if ~constrained
        % disable steering QP solves by increasing min_fallback_level.
        min_fallback_level  = max(  min_fallback_level,     ...
                                    POSTQP_FALLBACK_LEVEL   );
        % make sure max_fallback_level is at min_fallback_level
        max_fallback_level  = max(min_fallback_level, max_fallback_level);
    end
    if max_fallback_level > 0 
        APPLY_IDENTITY      = @(x) x;
    end
      
    if ~isempty(halt_log_fn)
        get_bfgs_state_fn = @bfgs_obj.getState;
        user_halt = halt_log_fn(0, x, penaltyfn_at_x, zeros(n,1),       ...
                                get_bfgs_state_fn, H_QP,                ...
                                1, 0, 1, stat_vec, stat_val, 0          );
    end

    if print_level
        printer.init(penaltyfn_at_x,stat_val,qps_solved);
    end

    rel_diff = inf;
    if converged()
        return
    elseif user_halt
        prepareTermination(3);
        return
    end
    
    % set up a more convenient function handles to reduce fixed arguments
    steering_fn     = @(penaltyfn_parts,H) qpSteeringStrategy(          ...
                            penaltyfn_parts,    H,                      ...
                            steering_l1_model,  steering_ineq_margin,   ...
                            steering_maxit,     steering_c_viol,        ...
                            steering_c_mu,      quadprog_opts           );

    linesearch_fn   = @(x,f,g,p,ls_maxit) linesearchWeakWolfe(          ...
                            x, f, g, p,                                 ...
                            @penaltyfn_obj.evaluatePenaltyFunction,     ...
                            wolfe1, wolfe2, fvalquit, ls_maxit, step_tol);
                                                      
    % we'll use a while loop so we can explicitly update the counter only
    % for successful updates.  This way, if the search direction direction
    % can't be used and the code falls back to alternative method to try
    % a new search direction, the iteration count is not increased for
    % these subsequent fallback attempts
    
    % loop control variables
    fallback_level      = min_fallback_level;
    random_attempts     = 0;
    iter                = 1;
    evals_so_far        = penaltyfn_obj.getNumberOfEvaluations();
    while iter <= maxit
             
%         Call standard steering strategy to produce search direction p
%         which hopefully "promotes progress towards feasibility".
%         However, if the returned p is empty, this means all QPs failed
%         hard.  As a fallback, steering will be retried with steepest
%         descent, i.e. H temporarily  set to the identity.  If this
%         fallback also fails hard, then the standard BFGS search direction
%         on penalty function is tried.  If this also fails, then steepest
%         will be tried.  Finally, if all else fails, randomly generated
%         directions are tried as a last ditch effort.
%         NOTE: min_fallback_level and max_fallback_level control how much
%               of this fallback range is available.
% 
%         NOTE: the penalty parameter is only lowered by two actions:
%         1) either of the two steering strategies lower mu and produce
%            a step accepted by the line search
%         2) a descent direction (generated via any fallback level) is not
%            initially accepted by the line search but a subsequent
%            line search attempt with a lowered penalty parameter does
%            produce an accepted step.
        
        penalty_parameter_changed = false;
        if fallback_level < POSTQP_FALLBACK_LEVEL  
            if fallback_level == 0
                apply_H_steer = apply_H_QP_fn;  % standard steering   
            else
                apply_H_steer = APPLY_IDENTITY; % "degraded" steering 
            end
            try
                [p,mu_new] = steering_fn(penaltyfn_at_x,apply_H_steer);
            catch err
                switch err.identifier
                    case 'GRANSO:steeringQuadprogFailure'
                        if ~halt_on_quadprog_error && bumpFallbackLevel()
                            if print_level > 2 
                                printer.qpError(iter,err,'STEERING');
                            end
                            continue      
                        else
                            err.rethrow();
                        end
                    otherwise
                        err.rethrow();
                end
            end
            penalty_parameter_changed = mu_new ~= mu;
            if penalty_parameter_changed 
                [f,g,mu] = penaltyfn_obj.updatePenaltyParameter(mu_new);
            end
        elseif fallback_level == 2
            p = -apply_H_fn(g);   % standard BFGS 
        elseif fallback_level == 3
            p = -g;     % steepest descent
        else
            p = randn(n,1);
            random_attempts = random_attempts + 1;
        end
               
        [p,is_descent,fallback_on_this_direction] = checkDirection(p,g);

        if fallback_on_this_direction
            if bumpFallbackLevel()
                continue    % try iteration again with new fallback
            else % all fallbacks have failed - quit!
                prepareTermination(9); % not a descent descent direction
                return
            end
        else % ATTEMPT LINE SEARCH
            f_prev = f;      % for relative termination tolerance
            g_prev = g;      % necessary for BFGS update
            if is_descent
                ls_procedure_fn = @linesearchDescent;
            else
                ls_procedure_fn = @linesearchNondescent;
            end
            % this will also update gprev if it lowers mu and it succeeds
            [alpha,x_new,f,g,linesearch_failed] = ls_procedure_fn(x,f,g,p);
        end
            
        if linesearch_failed
            % first get lowest mu attempted (restore will erase it)
            mu_lowest = penaltyfn_obj.getPenaltyParameter();
            % now, for all failure types, restore last accepted iterate
            can_fallback = bumpFallbackLevel();
            if linesearch_failed == 1  % bracketed minimizer but LS failed
                feasible = penaltyfn_obj.isFeasibleToTol();
                if halt_on_linesearch_bracket && feasible
                    prepareTermination(6);
                    return
                elseif can_fallback
                    continue
                else % return 6 (feasible) or 7 (infeasible)
                    prepareTermination(6 + ~feasible);
                    return
                end
            elseif linesearch_failed == 2  % couldn't bracket minimizer
                prepareTermination(8);
                return
            else % failed on nondescent direction
                if can_fallback
                    continue
                else
                    prepareTermination(9);
                end
            end
        end
        
        % ELSE LINE SEARCH SUCCEEDED - STEP ACCEPTED
        
        % compute relative difference of change in penalty function values
        % this will be infinity if previous value was 0 or if the value of
        % the penalty parameter was changed
        if penalty_parameter_changed || f_prev == 0
            rel_diff = inf;
        else
            rel_diff = abs(f - f_prev) / abs(f_prev);
        end
        
        % update x to accepted iterate from line search
        % mu is already updated by line search if lowered
        x = x_new;

        % Update all components of the penalty function evaluated
        % at the new accepted iterate x and snapsnot the data.
        penaltyfn_at_x          = penaltyfn_obj.snapShot();

        % for stationarity condition
        [   stat_vec,       ...
            stat_val,       ...
            qps_solved,     ...
            n_grad_samples  ]   = computeApproxStationarityVector();
            
        
        ls_evals = penaltyfn_obj.getNumberOfEvaluations()-evals_so_far;
        
        % Perform full or limited memory BFGS update
        % This computation is done before checking the termination
        % conditions because we wish to provide the most recent (L)BFGS
        % data to users in case they desire to restart.   
        applyBfgsUpdate(alpha,p,g,g_prev);
     
        if ~isempty(halt_log_fn)
            user_halt = halt_log_fn(iter, x, penaltyfn_at_x, p,         ...
                                    get_bfgs_state_fn, H_QP,            ...
                                    ls_evals, alpha, n_grad_samples,    ...
                                    stat_vec, stat_val, fallback_level  );
        end
       
        if print_level && mod(iter,print_frequency) == 0
            printer.iter(   iter,           penaltyfn_at_x,         ...
                            fallback_level, random_attempts,        ...
                            ls_evals,       alpha,                  ...
                            n_grad_samples, stat_val,   qps_solved  );     
        end
            
        % reset fallback level counters
        fallback_level  = min_fallback_level;
        random_attempts = 0;
        evals_so_far    = penaltyfn_obj.getNumberOfEvaluations();
  
        % check convergence/termination conditions
        if converged()
            return
        elseif user_halt
            prepareTermination(3);
            return
        elseif maxclocktime < inf && toc(t_start) > maxclocktime
            prepareTermination(5);
            return
        end
        
        % if cond(H) > regularize_limit, make a regularized version of H
        % for QP solvers to use on next iteration
        if iter < maxit     % don't bother if maxit has been reached
            [apply_H_QP_fn, H_QP] = get_apply_H_QP_fn();
        end

        iter = iter + 1; % only increment counter for successful updates
    end % while loop

    prepareTermination(4);  % max iterations reached

    % PRIVATE NESTED FUNCTIONS
    
    function [p,is_descent,fallback] = checkDirection(p,g)    
        fallback            = false;
        gtp                 = g'*p;
        if isnan(gtp) || isinf(gtp)
            is_descent      = false;
            fallback        = true;    
        else
            if gtp > 0 && fallback_level == LAST_FALLBACK_LEVEL
                % randomly generated ascent direction, flip sign of p
                p           = -p;
                is_descent  = true;
            else
                is_descent  = gtp < 0;
            end
            if ~is_descent && linesearch_nondescent_maxit == 0
                fallback    = true; 
            end
        end
    end

    function can_fallback = bumpFallbackLevel()
        penaltyfn_obj.restoreSnapShot();
       
        can_fallback        = fallback_level < max_fallback_level;
        if can_fallback
            fallback_level  = fallback_level + 1;
        elseif fallback_level == LAST_FALLBACK_LEVEL ...
                && random_attempts < max_random_attempts
            can_fallback    = true;
        end
    end

    % only try a few line search iterations if p is not a descent direction
    function [alpha, x, f, g, fail] = linesearchNondescent(x,f,g,p)
        [alpha,x,f,g,fail] = linesearch_fn( x,f,g,p,                    ...
                                            linesearch_nondescent_maxit );
        fail = 0 + 3*(fail > 0);
    end

    % regular weak Wolfe line search 
    % NOTE: this function may lower variable "mu" for constrained problems
    function [alpha, x_ls, f_ls, g_ls, fail] = linesearchDescent(x,f,g,p)
        
        % we need to keep around f and g so use _ls names for ls results
        ls_fn                       = @(f,g) linesearch_fn(x,f,g,p,inf);
        [alpha,x_ls,f_ls,g_ls,fail] = ls_fn(f,g);
                        
        % If the problem is constrained and the line search fails without 
        % bracketing a minimizer, it may be because the objective is 
        % unbounded below off the feasible set.  In this case, we can retry
        % the line search with progressively lower values of mu.   
        if constrained && fail == 2
        
            mu_ls = mu; % the original value of the penalty parameter
           
            if iter < 2 
                reattempts  = linesearch_reattempts_x0;
                ls_c_mu     = linesearch_c_mu_x0;
            else
                reattempts  = linesearch_reattempts;
                ls_c_mu     = linesearch_c_mu;
            end
              
            for j = 1:reattempts
                % revert to last good iterate (since line search failed)
                penaltyfn_obj.restoreSnapShot();
                % lower the trial line search penalty parameter further
                mu_ls       = ls_c_mu * mu_ls;
                [f,g]       = penaltyfn_obj.updatePenaltyParameter(mu_ls);
                gprev_ls    = g;
                
                if print_level > 1
                    printer.lineSearchRestart(iter,mu_ls);
                end
                
                [alpha,x_ls,f_ls,g_ls,failed_again] = ls_fn(f,g);
               
                if ~failed_again % LINE SEARCH SUCCEEDED 
                    % make sure mu and and gprev are up-to-date, since the
                    % penalty parameter has been lowered 
                    fail    = false;
                    mu      = penaltyfn_obj.getPenaltyParameter();
                    g_prev  = gprev_ls;
                    return
                end
            end
        end
       
        % LINE SEARCH EITHER SUCCEEDED OR FAILED
        % no need to restore snapshot if line search failed since 
        % bumpFallbackLevel() will be called and it requests the last  
        % snapshot to be restored.
    end

    function [  stat_vec,   ...
                stat_value, ...
                n_qps,      ...
                n_samples,  ...
                dist_evals  ] = computeApproxStationarityVector()
            
        % first check the smooth case (gradient of the penalty function).
        % If its norm is small, that indicates that we are at a smooth 
        % stationary point and we can return this measure and terminate
        stat_vec        = penaltyfn_at_x.p_grad;
        stat_value      = norm(stat_vec);
        if stat_value <= opt_tol
            n_qps       = 0;
            n_samples   = 1;
            dist_evals  = 0;
            penaltyfn_obj.addStationarityMeasure(stat_value);
            return
        end
      
        % otherwise, we must do a nonsmooth stationary point test
            
        % add new gradients at current iterate to cache and then get
        % all nearby gradients samples from history that are
        % sufficiently close to current iterate (within a ball of
        % radius evaldist centered at the current iterate x)
        [grad_samples,dist_evals] = get_nearby_grads_fn();
        
        % number of previous iterates that are considered sufficiently
        % close, including the current iterate
        n_samples = length(grad_samples);
        
        % nonsmooth optimality measure
        [stat_vec,n_qps,ME] = qpTerminationCondition(   penaltyfn_at_x, ...
                                                        grad_samples,   ...
                                                        apply_H_QP_fn,  ...
                                                        quadprog_opts   );
        stat_value = norm(stat_vec);
        penaltyfn_obj.addStationarityMeasure(stat_value);
        
        if print_level > 2 && ~isempty(ME)
            printer.qpError(iter,ME,'TERMINATION');
        end
    end

    function tf = converged()
        tf = true;
        % only converged if point is feasible to tolerance
        if penaltyfn_at_x.feasible_to_tol
            if stat_val <= opt_tol
                prepareTermination(0);
                return
            elseif rel_diff <= rel_tol
                prepareTermination(1);   
                return
            elseif penaltyfn_at_x.f <= fvalquit
                prepareTermination(2);
                return
            end
        end
        tf = false;
    end

    function prepareTermination(code)
        info.termination_code   = code;
        if code == 8 && constrained
            info.mu_lowest      = mu_lowest;
        end
    end

    function applyBfgsUpdate(alpha,p,g,gprev)
                    
        s               = alpha*p;
        y               = g - gprev;
        sty             = s'*y;
        
        if damping > 0
            [y,sty,damped] = bfgsDamping(damping,apply_H_fn,s,y,sty);
        end
        
        update_code     = bfgs_update_fn(s,y,sty,damped);
        
        if update_code > 0 && print_level > 1
            printer.bfgsInfo(iter,update_code);
        end
    end

    function [applyH, H] = getApplyH()
        applyH  = bfgs_obj.applyH;
        H       = [];
    end

    function [applyHr, Hr] = getApplyHRegularized()
        % This should only be called when running full memory BFGS as
        % getState() only returns the inverse Hessian as a dense matrix in
        % this case.  For L-BFGS, getState() returns a struct of data.
        [Hr,code] = regularizePosDefMatrix( bfgs_obj.getState(),        ...
                                            regularize_threshold,       ...
                                            regularize_max_eigenvalues  );
        if code == 2 && print_level > 2
            printer.regularizeError(iter);
        end
        applyHr  = @(x) Hr*x;
        
        % We only return Hr so that it may be passed to the halt_log_fn,
        % since (advanced) users may wish to look at it.  However, if
        % regularization was actually not applied, i.e. H = Hr, then we can
        % set Hr = [].  Users can already get H since @bfgs_obj.getState
        % is passed into halt_log_fn and the [] value will indicate to the 
        % user that regularization was not applied (which can be checked
        % more efficiently and quickly than comparing two matrices).   
        if code == 1
            Hr = [];    
        end
    end
end

function [grads,dist_evals] = getNearbyGradients(penaltyfn_obj,grad_nbd_fn)
    [f_grad, ci_grad, ce_grad] = penaltyfn_obj.getGradients();
    grads = struct('F', f_grad, 'CI', ci_grad, 'CE', ce_grad);     
    [~,grads,dist_evals] = grad_nbd_fn(penaltyfn_obj.getX(), grads);
end