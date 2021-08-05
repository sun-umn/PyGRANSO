function [  penalty_fn_object,  ...
            grad_norms_at_x0    ] = makePenaltyFunction(params,         ...
                                                        obj_fn,         ...
                                                        varargin        )
%   makePenaltyFunction: 
%       creates an object representing the penalty function for 
%           min obj_fn 
%           subject to ineq_fn <= 0
%                        eq_fn == 0
%       where the penalty function is specified with initial penalty 
%       parameter value mu and is applied to the objective function.
%       Roughly, this means:
%           mu*obj_fn   + sum of active inequality constraints 
%                       + sum of absolute value of eq. constraints
%
%   USAGE:
%   - obj_fn evaluates objective and constraints simultaneously:
%
%     makePenaltyFunction(opts,obj_fn);  
%   
%   - objective and inequality and equality constraints are all evaluated
%     separately by obj_fn, ineq_fn, and eq_fn respectively:
%
%     makePenaltyFunction(opts,obj_fn,ineq_fn,eq_fn); 
%   
%   The first usage may be both more convenient and efficient if computed 
%   value appear across the objectives and various constraints 
%   
%   INPUT:
%       params                      a struct of required parameters
% 
%       .x0                         [ n x 1 real vector ]
%           Initial point to evaluate the penalty function at.
%                         
%       .mu0                        [ nonnegative real value ]
%           Initial value of the penalty parameter.
% 
%       .prescaling_threshhold      [ positive real value ] 
%           Determines the threshold at which prescaling is applied to the
%           objective or constraint functions.  Prescaling is only applied
%           to the particular functions when the norms of their gradients
%           exceed the threshold; these functions are precaled so that the
%           norms of their gradients are equal to the prescaling_threshold.
%
%       .viol_ineq_tol              [ a nonnegative real value ]
%           A point is considered infeasible if the total violation of the 
%           inequality constraint function(s) exceeds this value.
% 
%       .viol_eq_tol                [ a nonnegative real value ]
%           A point is considered infeasible if the total violation of the 
%           equality constraint function(s) exceeds this value.
%                           
%       obj_fn                      [ function handle ]
%           This function takes a single argument, an n by 1 real vector, 
%           and evaluates either: 
%               Only the objective:  
%                   [f,grad] = obj_fn(x)
%               In this case, ineq_fn and eq_fn must be provided as
%               following input arguments.  If there are no (in)equality
%               constraints, (in)eq_fn should be set as [].
%
%               Or the objective and constraint functions together:
%                   [f,grad,ci,ci_grad,ce,ce_grad] = obj_fn(x)
%               In this case, ci and/or ce (and their corresponding
%               gradients) should be returned as [] if no (in)equality
%               constraints are given.
%
%       ineq_fn                     [ function handle ]
%           This function handle evaluate the inequality constraint 
%           functions and their gradients:
%               [ci,ci_grad] = ineq_fn(x)
%           This argument is required when the given obj_fn only evaluates 
%           the objective; this argument may be set to [] if there are no
%           inequality constraints.
%
%       ineq_fn                     [ function handle ]
%           This function handle evaluate the equality constraint functions
%           and their gradients:
%               [ce,ce_grad] = eq_fn(x)
%           This argument is required when the given obj_fn only evaluates 
%           the objective; this argument may be set to [] if there are no
%           inequality constraints.
%                               
%       NOTE: Each function handle returns the value of the function(s) 
%             evaluated at x, along with the corresponding gradient(s).  
%             If there are n variables and p inequality constraints, then
%             ci_grad should be an n by p matrix of p gradients for the p
%             inequality constraints.
%
%   OUTPUT:
%       p_obj       struct whose fields are function handles to manipulate 
%                   the penalty function object p_obj, with methods:
%
%           [p,p_grad,is_feasible] = p_obj.evaluatePenaltyFunction(x)
%           Evaluates the penalty function at vector x, returning its value
%           and gradient of the penalty function, along with a logical
%           indicating whether x is considered feasible (with respect to
%           the violation tolerances). 
%           NOTE: this function evaluates the underlying objective and
%           constraint functions at x.
%
%           [p,p_grad,mu_new] = p_obj.updatePenaltyParameter(mu_new)
%           Returns the updated value and gradient of the penalty function
%           at the current point, using new penalty parameter mu_new.  This
%           relies on using the last computed values and gradients for the
%           objective and constraints which are saved internally.
%
%           x = p_obj.getX();
%           Returns the current point that the penalty function has been
%           evaluated at.
% 
%           [f,f_grad,tv_l1,tv_l1_grad] = p_obj.getComponentValues()
%           Returns the components of the penalty function, namely the
%           current objective value and its gradient and the current l1
%           total violation and its gradient.
%
%           [p,p_grad] = p_obj.getPenaltyFunctionValue()
%           Returns the current value and gradient of the penalty function.
% 
%           mu = p_obj.getPenaltyParameter()
%           Returns the current value of the penalty parameter
%
%           [f_grad_out, ci_grad_out, ce_grad_out] = p_obj.getGradients()
%           Returns the gradients of the objective and constraints
%
%           n = p_obj.getNumberOfEvaluations()
%           Returns the number of times evaluatePenaltyFunction has been
%           called.
% 
%           [soln, stat_value] = p_obj.getBestSolutions()
%           Returns a struct of the best solution(s) encountered so far, 
%           along with an approximate measure of stationarity at the
%           current point.
%
%           tf = p_obj.isFeasibleToTolerances()
%           Returns true if the current point is considered feasible with
%           respect to the violation tolerances.
%
%           tf = p_obj.isPrescaled()
%           Returns true if prescaling was applied to any of the objective 
%           and/or constraint functions.
%       
%           tf = p_obj.hasConstraints()
%           Returns true if the specified optimization problem has
%           constraints.
%
%           n = p_obj.numberOfInequalities()
%           Returns the number of inequality functions.
% 
%           n = p_obj.numberOfEqualities()
%           Returns the number of equality functions.
%
%           s = p_obj.snapShot()
%           Takes a snap shot of the all current values related to the 
%           penalty function, stores it internally, and also returns it to 
%           the caller.  Allows the current state to be subsequently 
%           recalled if necessary.
% 
%           p_obj.restoreSnapShot()
%           Restores the state of the penalty function back to the last
%           time snapShot() was invoked (so we don't need to recompute it),
%           or, if passed snap shot data as an input argument, it will
%           restore back to the user-provided state.
%   
%           p_obj.addStationarityMeasure(value)
%           Add the stationarity measure value to the current state.
%
%       grad_norms_at_x0        
%           A struct of the norms of the gradients of objective and
%           constraint functions evaluated at x0.  The struct contains
%           fields .f, .ci, and .ce.  
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
%   makePenaltyFunction.m introduced in GRANSO Version 1.0.
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
     
    if ~isa(obj_fn, 'function_handle')
        error(  'GRANSO:userSuppliedFunctionsError',        ...
                'obj_fn must be a function handle of x.'    );
    end

    % local storage for function and gradients and the current x
    
    x = params.x0;
    n = length(x);
    
    % objective and its gradient
  
    if nargin < 4
        try 
            [f,f_grad,obj_fn,ineq_fn,eq_fn] = splitEvalAtX(obj_fn,x);
        catch err2           
            ME = MException('GRANSO:userSuppliedFunctionsError',        ...
                'failed to evaluate [f,grad,ci,ci_grad,ce,ce_grad] = obj_fn(x0).');
            ME = addCause(ME, err2);
            ME.throwAsCaller();
        end
    else
        try
            [f,f_grad] = obj_fn(x);
        catch err1
            ME = MException('GRANSO:userSuppliedFunctionsError',        ...
                            'failed to evaluate [f,f_grad] = obj_fn(x0).');
            ME = addCause(ME, err1);
            ME.throwAsCaller();
        end
        ineq_fn = varargin{1};
        eq_fn   = varargin{2};
    end
    assertFnOutputs(n,f,f_grad,'objective');
    
    prescaling_threshold = params.prescaling_threshold;
    % checking scaling of objective and rescale if necessary
    f_grad_norm = norm(f_grad);
    if f_grad_norm > prescaling_threshold
        scaling_f   = prescaling_threshold / f_grad_norm;
        obj_fn      = @(x) rescaleObjective(x,obj_fn,scaling_f);
        f           = f * scaling_f;
        f_grad      = f_grad * scaling_f;
    else
        scaling_f   = [];
    end
   
    % setup inequality and equality constraints, violations, and scalings
    
    [   eval_ineq_fn,           ...
        ci,ci_grad,             ...
        tvi,tvi_l1,tvi_l1_grad, ...
        ci_grad_norms,          ...
        scaling_ci,             ...
        ineq_constrained        ] = setupConstraint(x,                  ...
                                                    ineq_fn,            ...
                                                    @evalInequality,    ...
                                                    true,               ...
                                                    prescaling_threshold);
                                                
    [   eval_eq_fn,             ...
        ce,ce_grad,             ...
        tve,tve_l1,tve_l1_grad, ...
        ce_grad_norms,          ...
        scaling_ce,             ...
        eq_constrained          ] = setupConstraint(x,                  ...
                                                    eq_fn,              ...
                                                    @evalEquality,      ...
                                                    false,              ...
                                                    prescaling_threshold);
      
    grad_norms_at_x0 = struct(  'f',    f_grad_norm,    ...      
                                'ci',   ci_grad_norms,  ...
                                'ce',   ce_grad_norms   );
                            
    scalings        = [];
    if ~isempty(scaling_f)
        scalings.f  = scaling_f;
    end
    if ~isempty(scaling_ci)
        scalings.ci = scaling_ci;
    end
    if ~isempty(scaling_ce)
        scalings.ce = scaling_ce;
    end
    prescaled       = ~isempty(scalings);
    
    constrained = ineq_constrained || eq_constrained;
    if constrained
        mu                          = params.mu0;
        update_penalty_parameter_fn = @updatePenaltyParameter;
        viol_ineq_tol               = params.viol_ineq_tol;
        viol_eq_tol                 = params.viol_eq_tol;
        is_feasible_to_tol_fn       = @isFeasibleToTol;     
    else
        % unconstrained problems should have fixed mu := 1 
        mu                          = 1;
        update_penalty_parameter_fn = @penaltyParameterIsFixed;
        is_feasible_to_tol_fn       = @(varargin) true;
    end
    
    feasible_to_tol = is_feasible_to_tol_fn(tvi,tve);                                     
    tv              = max(tvi,tve);
    tv_l1           = tvi_l1 + tve_l1;
    tv_l1_grad      = tvi_l1_grad + tve_l1_grad;
    p               = mu*f + tv_l1;
    p_grad          = mu*f_grad + tv_l1_grad;
    
    % to be able to rollback to a previous iterate and mu, specifically
    % last time snapShot() was invoked
    fn_evals        = 1;
    snap_shot       = [];
    at_snap_shot    = false;
    stat_value      = nan;
    
    if constrained 
        unscale_fields_fn   = @unscaleFieldsConstrained;
        update_best_fn      = @updateBestSoFarConstrained;
        get_best_fn         = @getBestConstrained;
        most_feasible       = getInfoForXConstrained();
        best_to_tol         = [];
    else
        unscale_fields_fn   = @unscaleFields;
        update_best_fn      = @updateBestSoFar;
        get_best_fn         = @getBest;
        best_unconstrained  = getInfoForX();
    end
    
    update_best_fn();
    
    % output object with methods
    penalty_fn_object = struct(                                         ...
        'evaluatePenaltyFunction',      @evaluateAtX,                   ...
        'updatePenaltyParameter',       update_penalty_parameter_fn,    ...
        'getX',                         @getX,                          ...
        'getComponentValues',           @getComponentValues,            ...
        'getPenaltyFunctionValue',      @getPenaltyFunctionValue,       ...
        'getPenaltyParameter',          @getPenaltyParameter,           ...
        'getGradients',                 @getGradients,                  ...
        'getNumberOfEvaluations',       @getNumberOfEvaluations,        ...
        'getBestSolutions',             get_best_fn,                    ...
        'isFeasibleToTol',              @isFeasibleToTolerances,        ...
        'isPrescaled',                  @isPrescaled,                   ...
        'hasConstraints',               @() constrained,                ...
        'numberOfInequalities',         @() size(ci,1),                 ...
        'numberOfEqualities',           @() size(ce,1),                 ...
        'snapShot',                     @snapShot,                      ...
        'restoreSnapShot',              @restoreSnapShot,               ...
        'addStationarityMeasure',       @addStationarityMeasure         );
    
    % PUBLIC functions
    
    % evaluate objective, constraints, violation, and penalty function at x
    function [p_out,p_grad_out,feasible_to_tol_out] = evaluateAtX(x_in)
        
        try 
            at_snap_shot    = false;
            stat_value      = nan;
            fn_evals        = fn_evals + 1;
            % evaluate objective and its gradient
            [f,f_grad]      = obj_fn(x_in);
            % evaluate constraints and their violations (nested update)
            eval_ineq_fn(x_in); 
            eval_eq_fn(x_in);
        catch err3
            ME = MException('GRANSO:userSuppliedFunctionsError',        ...
                'failed to evaluate objective/constraint functions at x.');
            ME = addCause(ME, err3);
            ME.throwAsCaller();
        end
        
        x                   = x_in;
        feasible_to_tol     = is_feasible_to_tol_fn(tvi,tve);  
        tv                  = max(tvi,tve);
        tv_l1               = tvi_l1 + tve_l1;
        tv_l1_grad          = tvi_l1_grad + tve_l1_grad;
        p                   = mu*f + tv_l1;
        p_grad              = mu*f_grad + tv_l1_grad;
        
        % update best points encountered so far
        update_best_fn();
        
        % copy nested variables values to output arguments
        p_out               = p;
        p_grad_out          = p_grad;
        feasible_to_tol_out = feasible_to_tol;
    end

    function [f_o,f_grad_o,tv_l1_o,tv_l1_grad_o] = getComponentValues()
        f_o             = f;
        f_grad_o        = f_grad;
        tv_l1_o         = tv_l1;
        tv_l1_grad_o    = tv_l1_grad;
    end

    function tf = isFeasibleToTolerances()
        tf              = feasible_to_tol;
    end

    function evals = getNumberOfEvaluations()
        evals           = fn_evals;
    end
   
    % return the most recently evaluated x
    function [x_out] = getX()
        x_out           = x;
    end

    function [p_out,p_grad_out] = getPenaltyFunctionValue()
        p_out           = p;
        p_grad_out      = p_grad;
    end

    % update penalty function with new penalty parameter
    function [p_new,p_grad_new,mu_new] = updatePenaltyParameter(mu_new)
        mu              = mu_new;
        p               = mu*f + tv_l1;
        p_grad          = mu*f_grad + tv_l1_grad;
        p_new           = p;
        p_grad_new      = p_grad;     
    end

    % for unconstrained problems, ignore updates to mu
    function [p_new,p_grad_new,mu_new] = penaltyParameterIsFixed(varargin)
        mu_new          = mu;
        p_new           = p;
        p_grad_new      = p_grad;     
    end

    function mu_out = getPenaltyParameter()
        mu_out          = mu;
    end

    function [f_grad_out, ci_grad_out, ce_grad_out] = getGradients()
        f_grad_out      = f_grad;
        ci_grad_out     = ci_grad;
        ce_grad_out     = ce_grad;
    end

    function tf = isPrescaled()
        tf              = prescaled;
    end

    function addStationarityMeasure(stationarity_measure)
        stat_value = stationarity_measure;
        if at_snap_shot
            snap_shot.stat_value = stationarity_measure;
        end
    end

    % PRIVATE helper functions 
       
    function s = snapShot()
        % scalings never change so no need to snapshot them
        snap_shot = struct(                                             ...
                        'f',        f,      'f_grad',       f_grad,     ...
                        'ci',       ci,     'ci_grad',      ci_grad,    ...
                        'ce',       ce,     'ce_grad',      ce_grad,    ...
                        'tvi',      tvi,    'tve',          tve,        ...
                        'tv',       tv,                                 ...
                        'tvi_l1',   tvi_l1, 'tvi_l1_grad',  tvi_l1_grad,...
                        'tve_l1',   tve_l1, 'tve_l1_grad',  tve_l1_grad,...
                        'tv_l1',    tv_l1,  'tv_l1_grad',   tv_l1_grad, ...
                        'p',        p,      'p_grad',       p_grad,     ...
                        'mu',       mu,     'x',            x,          ...
                        'feasible_to_tol',  feasible_to_tol,            ...
                        'stat_value',       stat_value                  );
        s = snap_shot;
        at_snap_shot = true;
    end

    function s = restoreSnapShot(user_snap_shot)
        if nargin > 0
            s = user_snap_shot;
        else
            s = snap_shot;
        end
        
        if ~isempty(s)
            f               = s.f;
            f_grad          = s.f_grad;
            ci              = s.ci;
            ci_grad         = s.ci_grad;
            ce              = s.ce;
            ce_grad         = s.ce_grad;
            tvi             = s.tvi;
            tve             = s.tve;
            tv              = s.tv;
            tvi_l1          = s.tvi_l1;
            tvi_l1_grad     = s.tvi_l1_grad;
            tve_l1          = s.tve_l1;
            tve_l1_grad     = s.tve_l1_grad;
            tv_l1           = s.tv_l1;
            tv_l1_grad      = s.tv_l1_grad;
            p               = s.p;
            p_grad          = s.p_grad;
            mu              = s.mu;
            x               = s.x;
            feasible_to_tol = s.feasible_to_tol;
            stat_value      = s.stat_value;
            snap_shot       = s;
            at_snap_shot    = true;
        end   
    end
   
    function evalInequality(x,fn)
        [ci,ci_grad]                = fn(x);
        [tvi,tvi_l1,tvi_l1_grad]    = totalViolationInequality(ci,ci_grad);
    end

    function evalEquality(x,fn)
        [ce,ce_grad]                = fn(x);
        [tve,tve_l1,tve_l1_grad]    = totalViolationEquality(ce,ce_grad);
    end

    function tf = isFeasibleToTol(tvi,tve)
        % need <= since tolerances could be 0 for very demanding users ;-)
        tf = (tvi <= viol_ineq_tol && tve <= viol_eq_tol);
    end

    function s = getInfoForX()
        s = dataStruct(x,f);      
    end

    function s = getInfoForXConstrained()   
        s = dataStructConstrained(x,f,ci,ce,tvi,tve,tv,feasible_to_tol,mu);
    end

    function updateBestSoFar()
        if f < best_unconstrained.f
            best_unconstrained = getInfoForX();
        end       
    end

    function updateBestSoFarConstrained()      
        % Update the iterate which is closest to the feasible region.  In
        % the case of ties, keep the one that most minimizes the objective.
        update_mf   =   tv < most_feasible.tv || ...
                        (tv == most_feasible.tv && f < most_feasible.f);
        
        % Update iterate which is feasible w.r.t violation tolerances and
        % most minimizes the objective function
        update_btt  =   feasible_to_tol &&  ...
                        (isempty(best_to_tol) || f < best_to_tol.f);
        
        if update_mf || update_btt
            soln = getInfoForXConstrained();
            if update_mf
                most_feasible   = soln;
            end
            if update_btt
                best_to_tol     = soln;
            end
        end
    end

    function unscaled = unscaleFields(data)
        unscaled    = dataStruct(data.x,unscaleValues(data.f,scaling_f));
    end

    function unscaled = unscaleFieldsConstrained(data)
        f_u         = unscaleValues(data.f,scaling_f);
        ci_u        = unscaleValues(data.ci,scaling_ci);
        ce_u        = unscaleValues(data.ce,scaling_ce);
        tvi_u       = totalViolationMax(violationsInequality(ci_u));
        tve_u       = totalViolationMax(violationsEquality(ce_u));
        tv_u        = max(tvi_u,tve_u);
        unscaled    = dataStructConstrained(                            ...
                            data.x, f_u, ci_u, ce_u, tvi_u, tve_u, tv_u,...
                            is_feasible_to_tol_fn(tvi_u,tve_u), data.mu );
    end
    
    function [soln, stat_value_o] = getBest()
        final_field         = {'final', getInfoForX()};
        best_field          = {'best', best_unconstrained};
        if prescaled
            scalings_field  = getScalings();
            final_unscaled  = getUnscaledData(final_field);
            best_unscaled   = getUnscaledData(best_field);
        else
            scalings_field  = {};
            final_unscaled  = {};
            best_unscaled   = {};
        end
        soln = struct(  scalings_field{:},              ...
                        final_field{:},                 ...
                        final_unscaled{:},              ...
                        best_field{:},                  ...
                        best_unscaled{:}                );
        stat_value_o = stat_value;
    end

    function [soln, stat_value_o] = getBestConstrained()
        final_field         = {'final', getInfoForXConstrained();};
        feas_field          = {'most_feasible', most_feasible};
        if isempty(best_to_tol)
            best_field      = {};
        else
            best_field      = {'best', best_to_tol};           
        end
        if prescaled
            scalings_field  = getScalings();
            final_unscaled  = getUnscaledData(final_field);
            feas_unscaled   = getUnscaledData(feas_field);
            best_unscaled   = getUnscaledData(best_field);
        else
            scalings_field  = {};
            final_unscaled  = {};
            feas_unscaled   = {};
            best_unscaled   = {};
        end  
        soln = struct(  scalings_field{:},              ...
                        final_field{:},                 ...
                        final_unscaled{:},              ...
                        best_field{:},                  ...
                        best_unscaled{:},               ...
                        feas_field{:},                  ...
                        feas_unscaled{:}                );
        stat_value_o = stat_value;
    end

    function scalings_field = getScalings()
        scalings_field      = {'scalings',scalings};
    end

    function unscaled_data = getUnscaledData(data_field)
        if isempty(data_field)
            unscaled_data   = {};
        else
            [name,data]     = deal(data_field{1},data_field{2});
            unscaled_data   = {[name '_unscaled'],unscale_fields_fn(data)};
        end
    end

end

function assertFnOutputs(n,f,g,fn_name)
    if fn_name(1) == 'o'
        [arg1,arg2] = deal('function value','gradient');
        assertFn(isscalar(f),arg1,fn_name,'be a scalar');
        [r,c] = size(g);
        assertFn(c == 1,arg2,fn_name,'be a column vector');
    else
        [arg1,arg2] = deal('function value(s)','gradient(s)');
        [nf,c] = size(f);
        assertFn(nf >= 1 && c == 1,arg1,fn_name,'be a column vector');
        [r,ng] = size(g);
        assertFn(nf == ng,'number of gradients',fn_name,            ...
            'should match the number of constraint function values' );
    end

    assertFn(r == n,arg2,fn_name,                                   ...
        'have dimension matching the number of variables'           );

    assertFn(isRealValued(f),arg1,fn_name,'should be real valued');
    assertFn(isRealValued(g),arg2,fn_name,'should be real valued');
    assertFn(isFiniteValued(f),arg1,fn_name,'should be finite valued');
    assertFn(isFiniteValued(g),arg2,fn_name,'should be finite valued');
end

function assertFn(cond,arg_name,fn_name,msg)
    assert( cond,                                                       ...
            'GRANSO:userSuppliedFunctionsError',                        ...
            'The %s at x0 returned by the %s function should %s!',      ...
            arg_name,fn_name,msg                                        );
end

function [f,f_grad,obj_fn,ineq_fn,eq_fn] = splitEvalAtX(eval_at_x_fn,x0)
    
    [f,f_grad,ci,ci_grad,ce,ce_grad] = eval_at_x_fn(x0);
  
    obj_fn      = @objective;
    ineq_fn     = ternOp(isempty(ci), [], @inequality);
    eq_fn       = ternOp(isempty(ce), [], @equality);
    
    function [f,g] = objective(x)
        [f,g,ci,ci_grad,ce,ce_grad] = eval_at_x_fn(x);
    end

    function [c,c_grad] = inequality(varargin)
        c       = ci;
        c_grad  = ci_grad;
    end

    function [c,c_grad] = equality(varargin)
        c       = ce;
        c_grad  = ce_grad;
    end
end

function [vi,violated_indx] = violationsInequality(ci)
    vi                  = ci;
    violated_indx       = ci >= 0;
    vi(~violated_indx)  = 0;
end

function [ve,violated_indx] = violationsEquality(ce)
    ve                  = abs(ce);
    violated_indx       = ce >= 0;   % indeed, this is all of them 
end

function v_max = totalViolationMax(v)
    if isempty(v)
        v_max = 0;
    else
        v_max = max(v);
    end
end

function [tvi,tvi_l1,tvi_l1_grad] = totalViolationInequality(ci,ci_grad)
    [vi,indx]   = violationsInequality(ci);
    
    % l_inf penalty term for feasibility measure
    tvi         = totalViolationMax(vi);
    
    % l_1 penalty term for penalty function
    tvi_l1      = sum(vi);
    tvi_l1_grad = sum(ci_grad(:,indx),2);
end

function [tve,tve_l1,tve_l1_grad] = totalViolationEquality(ce,ce_grad)
    [ve,indx]   = violationsEquality(ce);

    % l_inf penalty term for feasibility measure
    tve         = totalViolationMax(ve);
    
    % l_1 penalty term for penalty function
    tve_l1      = sum(ve);
    tve_l1_grad = sum(ce_grad(:,indx),2) - sum(ce_grad(:,~indx),2);
end

function [f,g] = rescaleObjective(x,fn,scaling)
    [f,g]   = fn(x);
    f       = f*scaling;
    g       = g*scaling;
end

function [f,g] = rescaleConstraint(x,fn,scalings)
    [f,g]   = fn(x);
    f       = f.*scalings;
    g       = g*diag(scalings);
end

function values = unscaleValues(values,scalars)
    if ~isempty(scalars)
        values = values ./ scalars;
    end
end

function [  eval_fn,                                                ...
            c, c_grad,                                              ...
            tv, tv_l1, tv_l1_grad,                                  ...
            c_grad_norms,                                           ...
            scalings,                                               ...
            constrained] = setupConstraint( x0, c_fn, eval_fn,      ...
                                            inequality_constraint,  ...
                                            prescaling_threshold    )
                                        
    n = length(x0);
                            
    % eval_fn is either a function handle for evaluateInequality or
    % evaluateEquality so we can detect which we have based on its length
    if inequality_constraint
        viol_fn     = @totalViolationInequality;
        type_str    = 'in';
    else
        viol_fn     = @totalViolationEquality;
        type_str    = '';
    end

    scalings                = []; % default if no prescaling is applied
    if isempty(c_fn)
        eval_fn             = @(x)[];
        % These must have the right dimensions for computations to be 
        % done even if there are no such constraints
        c                   = zeros(0,1);
        c_grad              = zeros(length(x0),0);
        c_grad_norms        = 0;
        tv                  = 0;
        tv_l1               = 0;
        tv_l1_grad          = 0;
        constrained         = false;
    elseif isa(c_fn,'function_handle')
        try 
            [c,c_grad]      = c_fn(x0);
        catch err3
            ME = MException('GRANSO:userSuppliedFunctionsError',        ...
                'failed to evaluate [c,c_grad] = %seq_fn(x0).',type_str);
            ME = addCause(ME, err3);
            ME.throwAsCaller();                
        end
        assertFnOutputs(n,c,c_grad,[type_str 'equality constraints']);    
        c_grad_norms        = sum(c_grad.^2,1).^0.5;
        % indices of gradients whose norms are larger than limit
        indx                = c_grad_norms > prescaling_threshold;
        if any(indx)
            scalings        = ones(length(c),1);
            % we want to rescale these "too large" functions so that 
            % the norms of their gradients are set to limit at x0
            scalings(indx)  = prescaling_threshold ./ c_grad_norms(indx);
            c_fn            = @(x) rescaleConstraint(x,c_fn,scalings);
            % rescale already computed constraints and gradients
            c               = c .* scalings;
            c_grad          = c_grad * diag(scalings);
        end
        [tv,tv_l1,tv_l1_grad]   = viol_fn(c,c_grad);
        % reset eval_fn so that it computes the values and gradients 
        % for both the constraint and the corresponding violation
        eval_fn             = @(x) eval_fn(x,c_fn);
        constrained         = true;
    else       
        error('GRANSO:userSuppliedFunctionsError',                  ...
              ['%seq_fn must be a function handle of x or empty, '  ...
               'that is, [].\n'],type_str); 
    end
end

function s = dataStruct(x,f)
    s = struct('x',x,'f',f);
end

function s = dataStructConstrained(x,f,ci,ce,tvi,tve,tv,feasible_to_tol,mu)
    s = struct( 'x',                x,                  ...
                'f',                f,                  ...
                'ci',               ci,                 ...
                'ce',               ce,                 ...
                'tvi',              tvi,                ...
                'tve',              tve,                ...
                'tv',               tv,                 ...
                'feasible_to_tol',  feasible_to_tol,    ...
                'mu',               mu                  );
end