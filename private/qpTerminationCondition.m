function [d,qps_solved,ME] = qpTerminationCondition(penaltyfn_at_x,     ...
                                                    gradient_samples,   ...
                                                    apply_Hinv,         ...
                                                    quadprog_options    )
%   qpTerminationCondition:
%       computes the smallest vector in the convex hull of gradient samples
%       provided in cell array gradient samples, given the inverse Hessian
%       (or approximation to it)
%
%   INPUT:
%       penaltyfn_at_x      struct containing fields for:
%       .mu                 current penalty parameter
%       .f                  objective function evaluated at x
%       .ci                 inequality constraints evaluated at x 
%       .ce                 equality constraints evaluated at x
%
%       gradient_samples    a cell array of structs containing fields
%                           'F', 'CI', and 'CE', which contain the
%                           respective gradients for a history or 
%                           collection of the objective
%                           function, the inequality constraints, and
%                           equality constraints, evaluated at different 
%                           x_k.  Each index of the cell array contains 
%                           these F, CI, and CE values for a different
%                           value of x_k.  One of these x_k should
%                           correspond to x represented in penaltyfn_parts
%
%       apply_Hinv          function handle apply_Hinv(x)
%                           returns b = Hinv*x where Hinv is the inverse
%                           Hessian (or approximation to it).  Hinv must be
%                           positive definite.  x may be a single column or
%                           matrix.
% 
%       quadprog_opts       struct of options for quadprog interface
%                           It must be provided but it may be set as []
%
%   OUTPUT:
%       d                   smallest vector in convex hull of gradient 
%                           samples d is a vector of Infs if QP solver 
%                           fails hard
%   
%       qps_solved          number of QPs attempted in order to compute
%                           some variant of d.  If quadprog fails when
%                           attempting to compute d, up to three
%                           alternative QP formulations will be attempted.
%
%       ME                  empty [] if default d was computed normally.
%                           Otherwise an MException object containing
%                           the causes of why each method of computing d 
%                           failed, accrued in the .cause field.
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
%   qpTerminationCondition.m introduced in GRANSO Version 1.0.
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
                           
    mu          = penaltyfn_at_x.mu;
    l           = length(gradient_samples);
    p           = l * length(penaltyfn_at_x.ci);
    q           = l * length(penaltyfn_at_x.ce);                                
   
    F           = penaltyfn_at_x.f * ones(l,1); 
    CI          = repmat(penaltyfn_at_x.ci,l,1);
    CE          = repmat(penaltyfn_at_x.ce,l,1);
    
    % convert cell array fields F, CI, CE to struct array with same
    grads_array = cell2mat(gradient_samples);
    % convert struct array into individual arrays of samples
    F_grads     = [grads_array.F];      % n by l
    CI_grads    = [grads_array.CI];     % n by p
    CE_grads    = [grads_array.CE];     % n by q 
   
    % Set up arguments for quadprog interface
    all_grads   = [CE_grads F_grads CI_grads];
    Hinv_grads  = apply_Hinv(all_grads);
    H           = all_grads' * Hinv_grads;
    % Fix H since numerically, it is unlikely to be _perfectly_ symmetric 
    H           = (H + H') / 2;
    f           = -[CE; F; CI];
    LB          = [-ones(q,1); zeros(l+p,1)];
    UB          = [ones(q,1); mu*ones(l,1); ones(p,1)];
    Aeq         = [zeros(1,q) ones(1,l) zeros(1,p)];
    beq         = mu;
    
    solveQP_fn = @(H) solveQP(H,f,[],[],Aeq,beq,LB,UB,[],quadprog_options);
    
    [y,~,qps_solved,ME] = solveQPRobust();
      
    % If the QP solve(s) failed, return infinite vector so it can't
    % possibly trigger BFGS-SQP's convergence criteria
    if isempty(y)
        % its length is equal to the number of variables
        d = inf*ones(size(Hinv_grads,1),1); 
    else
        d = -Hinv_grads*y;
    end
    
    function [x,lambdas,stat_type,ME] = solveQPRobust()
       
        x       = [];
        lambdas = [];
        ME      = [];
        
        % Attempt to solve QP
        try 
            stat_type   = 1;
            [x,lambdas] = solveQP_fn(H);
            return
        catch err       
            ME = MException(                                            ...
                'GRANSO:terminationQP',                                 ...
                'Default stationarity measure could not be computed.'   );
            ME = ME.addCause(err);
        end       
           
        % QP solver failed, possibly because H was numerically nonconvex,
        % i.e. H may have tiny negative eigenvalues close to zero because
        % of rounding errors
       
        % Fall back strategy #1: replace Hinv with identity and try again
        try 
            stat_type   = 2;
            R           = all_grads' * all_grads;
            R           = (R+R')/2;
            [x,lambdas] = solveQP_fn(R);
            return
        catch err
            ME = ME.addCause(err);
        end  
        
        % Fall back strategy #2: revert to MATLAB's quadprog, if user is
        % using a different quadprog solver and reattempt with original H
        if ~isDefaultQuadprog() 
            users_paths = path;
            % put MATLAB's quadprog on top of path so that it is called
            addpath(getDefaultQuadprogPath());
            try 
                stat_type   = 3;
                [x,lambdas] = solveQP_fn(H);  
            catch err
                ME = ME.addCause(err);
            end
            % restore user's original list of paths (and their order!)
            path(users_paths); 
            
            % solve succeeded
            if ~isempty(x)
                return
            end
        end    
        
        % Fall back strategy #3: regularize H - this could be expensive
        % Even though min(eig(Hreg)) may still be tiny negative number,
        % this mild regularization seems to often be sufficient prevent 
        % MOSEK from complaining about nonconvexity and aborting. 
        try 
            stat_type       = 4;
            [V,D]           = eig(H);
            dvec            = diag(D);
            dvec(dvec < 0)  = 0;      
            Hreg            = V*diag(dvec)*V';  
            [x,lambdas]     = solveQP_fn(Hreg);
        catch err
            ME = ME.addCause(err);
        end
    end  
end

                                        