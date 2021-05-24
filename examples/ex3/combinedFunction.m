function [f,f_grad,ci,ci_grad,ce,ce_grad] = combinedFunction(w,x)
%   combinedFunction: (examples/ex3)
%       Defines a simple 2-variable nonsmooth Rosenbrock objective function
%       (to be minimized), subject to an inequality constraint, along with 
%       their respective gradients.  
% 
%       The format of this function adheres to GRANSO's combined format,
%       where all values are computed by a single function (which allows
%       for easy sharing of any data and intermediate computations between
%       the various functions). 
%       
%       GRANSO's convention is that the inequality constraints must be less
%       than or equal to zero to be satisfied.  For equality constraints,
%       GRANSO's convention is that they must be equal to zero.
%
%       GRANSO requires that gradients are column vectors.  For example, if
%       there was a second inequality constraint, its gradient would be
%       stored in the 2nd column of ci_grad.
%
%   USAGE:
%       [f,f_grad,ci,ci_grad,ce,ce_grad] = combinedFunction(w,x);
% 
%   INPUT:  
%       w                   constant for nonsmooth Rosenbrock function 
%                           real finite scalar value
%
%       x                   optimization variables
%                           real-valued column vector, 2 by 1 
%   
%   OUTPUT:
%       f                   value of the objective function at x
%                           scalar real value
% 
%       f_grad              gradient of the objective function at x.
%                           real-valued column vector, 2 by 1
% 
%       ci                  values of the inequality constraints at x.
%                           real-valued column vector, 2 by 1, where the
%                           jth entry is the value of jth constraint
%                 
%       ci_grad             gradient of the inequality constraint at x.
%                           real-valued matrix, 2 by 2, where the jth
%                           column is the gradient of the jth constraint
% 
%       ce                  value of the equality constraint(s) at x.
%                           [], since there aren't any equality constraints
%
%       ce_grad             gradient(s) of the equality constraint(s) at x.
%                           [], since there aren't any equality constraints
% 
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   examples/ex3/combinedFunction.m introduced in GRANSO Version 1.5.
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

    % OBJECTIVE VALUE AT X
    % nonsmooth Rosenbrock example with constant w 
    f           = w*abs(x(1)^2 - x(2)) + (1 - x(1))^2;
   
    % OBJECTIVE GRADIENT AT X
    % Compute the 2nd term
    f_grad      = [-2*(1-x(1)); 0];
    % Add in the 1st term, where we must handle the sign due to the 
    % absolute value
    if x(1)^2 - x(2) >= 0
        f_grad  = f_grad + w*[ 2*x(1); -1];
    else
        f_grad  = f_grad + w*[-2*x(1);  1];
    end

    
    % INEQUALITY CONSTRAINTS
    ci      = [sqrt(2)*x(1); 2*x(2)] - 1;

    % GRADIENTS OF THE TWO INEQUALITY CONSTRAINTS
    ci_grad = [ [sqrt(2); 0] [0; 2] ];
    
    
    % EQUALITY CONSTRAINT
    % Return [] when (in)equality constraints are not present.
    
    ce      = [];
    ce_grad = [];

end