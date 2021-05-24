function [ci,ci_grad] = inequalityConstraint(x)
%   inequalityConstraint: (examples/ex1)
%       Encodes a simple 2-variable inequality constraint and its gradient.  
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
%       [ci,ci_grad] = inequalityConstraint(x);
% 
%   INPUT:
%       x           optimization variables
%                   real-valued column vector, 2 by 1 
%   
%   OUTPUT:
%       ci          values of the inequality constraints at x.
%                   real-valued column vector, 2 by 1, where the jth entry 
%                   is the value of jth constraint
%                 
%       ci_grad     gradient of the inequality constraint at x.
%                   real-valued matrix, 2 by 2, where the jth column is the
%                   gradient of the jth constraint
% 
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   examples/ex1/inequalityConstraint.m introduced in GRANSO Version 1.5.
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

    % INEQUALITY CONSTRAINTS
    ci      = [sqrt(2)*x(1); 2*x(2)] - 1;

    % GRADIENTS OF THE TWO INEQUALITY CONSTRAINTS
    ci_grad = [ [sqrt(2); 0] [0; 2] ];
    
end