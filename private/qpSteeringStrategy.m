function [d,mu,reduction] = qpSteeringStrategy( penaltyfn_at_x,         ...
                                                apply_Hinv,             ...
                                                l1_model,               ...
                                                ineq_margin,            ...
                                                maxit,                  ...
                                                c_viol,                 ...
                                                c_mu,                   ...
                                                quadprog_options        )
%   qpSteeringStrategy:
%       attempts to find a search direction which promotes progress towards
%       feasibility.  
%
%   INPUT:
%       penaltyfn_at_x  struct containing fields for:
%       .mu             current penalty parameter
%       .f_grad         gradient of objective function at x
%       .ci             inequality constraints evaluated at x 
%       .ci_grad        corresponding gradients at x
%       .ce             equality constraints evaluated at x
%       .ce_grad        corresponding gradients at x
%       .tv_l1          total violation value at x (one norm)
%       .tv             total violation value at x (infinity norm)
%
%       l1_model        logical: determines whether or not the one norm 
%                       (the standard choice) or the infinity norm is used
%                       for the total violation measure, which affects the
%                       predicted violation reduction.
%   
%       ineq_margin     real value in [0,inf] setting the margin of 
%                       feasibility for problems having only inequality 
%                       constraints.  In this case, steering is selectively 
%                       disabled when the inequality constraints are all at 
%                       least ineq_margin away from being active.  Setting 
%                       ineq_margin to zero means that steering will only 
%                       be applied when one or more inequality constraints 
%                       are active ( >= 0).  Setting ineq_margin to inf 
%                       means that steering will be applied on every 
%                       iteration.  NOTE: this parameter has no effect if 
%                       equality constraints are present.
%
%       apply_Hinv      function handle apply_Hinv(x)
%                       returns b = Hinv*x where Hinv is the inverse
%                       Hessian (or approximation to it).  Hinv must be
%                       positive definite.  x may be a single column or
%                       matrix.
% 
%       maxit           max iteration count to try lowering penalty 
%                       parameter mu in order to find a good search 
%                       direction
%
%       c_viol          percentage of total violation needed to be acheived
%                       by the predicted reduction for a candidate 
%                       direction must be in (0,1)
%
%       c_mu            scalar factor to reduce penalty paremeter mu on 
%                       each iterative if resulting direction is not 
%                       acceptable must be in (0,1)
%
%       quadprog_opts   struct of options for quadprog interface.
%                       It must be provided but it may be set as []
%
%   OUTPUT:
%       d               candidate search direction
%                       d will be set to [] if all QP solves fail hard
%
%       mu              possibly lower value of the penalty parameter 
% 
%       reduction       amount of total violation reduction d is predicted 
%                       to yield via the linearized constraint model
%
%   THROWS:
%       error           if any call to quadprog either throws an error
%                       (which will be set as .cause of the GRANSO error)
%                       or if quadprog returns without error but its answer 
%                       is numerically invalid (e.g. inf, nan, empty, zero)
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
%   qpSteeringStrategy.m introduced in GRANSO Version 1.0.
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

    mu                  = penaltyfn_at_x.mu;
    f_grad              = penaltyfn_at_x.f_grad;
    ineq                = penaltyfn_at_x.ci;
    ineq_grad           = penaltyfn_at_x.ci_grad;
    eq                  = penaltyfn_at_x.ce;
    eq_grad             = penaltyfn_at_x.ce_grad;
    if l1_model
        predictedViolFn = @predictedViolationReductionL1;
        violation       = penaltyfn_at_x.tv_l1;
    else
        predictedViolFn = @predictedViolationReduction;
        violation       = penaltyfn_at_x.tv;
    end
    
    n_ineq              = length(ineq);
    n_eq                = length(eq);
    
    Hinv_f_grad         = apply_Hinv(f_grad);
    
    % factor to allow for some inaccuracy in the QP solver
    violation_tol       = sqrt(eps)*max(violation,1);
     
    % Set up arguments for quadprog interface
    c_grads             = [eq_grad ineq_grad];
    Hinv_c_grads        = apply_Hinv(c_grads);
    H                   = c_grads' * Hinv_c_grads;
    % Fix H since numerically, it is unlikely to be _perfectly_ symmetric 
    H                   = (H + H') / 2;
    mu_Hinv_f_grad      = mu * Hinv_f_grad;
    f                   = c_grads' * mu_Hinv_f_grad - [eq; ineq];
    LB                  = [-ones(n_eq,1); zeros(n_ineq,1)];
    UB                  = ones(n_eq + n_ineq, 1);
   
    % Check predicted violation reduction for search direction
    % given by current penalty parameter
    d                   = solveSteeringDualQP();
    reduction           = predictedViolFn(d);
    if reduction >= c_viol*violation - violation_tol
        return
    end
   
    % Disable steering if all inequality constraints are strictly 
    % feasible, i.e., at least ineq_margin away from the feasible boundary,
    % and no equality constraints are present.
    % Explicitly check for infinity in case ineq contains -inf 
    if ~isinf(ineq_margin) && ~any(ineq > -ineq_margin) && n_eq == 0
        return
    end
        
    % Predicted violation reduction was inadequate.  Check to see
    % if reduction is an adequate fraction of the predicted reduction 
    % when using the reference direction (given by the QP with the 
    % objective removed, that is, with the penalty parameter temporarily 
    % set to zero)
    updateSteeringQP(0);
    d_reference         = solveSteeringDualQP();
    reduction_reference = predictedViolFn(d_reference);
    if reduction >= c_viol*reduction_reference - violation_tol
        return
    end
   
    % iteratively lower penalty parameter to produce new search directions
    % which should hopefully have predicted reductions that are larger 
    % fractions of the predicted reduction of the reference direction
    for j = 1:maxit
        mu = c_mu * mu;
        updateSteeringQP(mu);
        d               = solveSteeringDualQP();
        reduction       = predictedViolFn(d);
        % Test new step's predicted reduction against reference's
        if reduction >= c_viol*reduction_reference - violation_tol
            return % predicted reduction is acceptable
        end
    end
    
    % All d failed to meet acceptable predicted violation reduction so 
    % just use the last search direction d produced for the lowest penalty 
    % parameter value considered.  
    
    
    % private helper functions
    
    % calculate predicted violation reduction for search direction d  using 
    % a linearized constraint model
    
    % l1 total violation
    function dL = predictedViolationReductionL1(d)
        dL = violation                                                  ...
                - sum(max([ineq + ineq_grad.'*d,zeros(n_ineq,1)],[],2)) ...
                - norm(eq + eq_grad.'*d,1);
    end

    % l-infinity total violation
    function dL = predictedViolationReduction(d)
        dL = violation - max(max(ineq + ineq_grad.'*d),0) ...
                       - norm(eq + eq_grad.'*d,inf);
    end
    
    % solve dual of steering QP to yeild new search direction
    % throws an error if QP solver failed somehow
    function d = solveSteeringDualQP()
        try 
            y = solveQP(H,f,[],[],[],[],LB,UB,[],quadprog_options); 
        catch err
            ME = MException(                                            ...
                'GRANSO:steeringQuadprogFailure',                       ...
                'Steering aborted due to a quadprog failure.'           );
            ME = ME.addCause(err);
            ME.throw();
        end
        d = -mu_Hinv_f_grad - (Hinv_c_grads * y);
    end
    
    % update penalty parameter dependent values for QP
    function updateSteeringQP(mu)
        mu_Hinv_f_grad  = mu * Hinv_f_grad;
        f               = c_grads' * mu_Hinv_f_grad - [eq; ineq];
    end
end