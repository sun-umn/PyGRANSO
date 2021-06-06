function [f,f_grad] = objectiveFunction(x)
%   objectiveFunction: (examples/ex1)
%       Encodes a simple objective function and its gradient.
%       
%       GRANSO will minimize the objective function.  The gradient must be
%       a column vector.
% 
%   USAGE:
%       [f,f_grad] = objectiveFunction(x);
% 
%   INPUT:
%       x           optimization variables
%                   real-valued column vector, 2 by 1 
%   
%   OUTPUT:
%       f           value of the objective function at x
%                   scalar real value
% 
%       f_grad      gradient of the objective function at x.
%                   real-valued column vector, 2 by 1
% 
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   examples/ex1/objectiveFunction.m introduced in GRANSO Version 1.5.
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
    % nonsmooth Rosenbrock example with constant 8 embedded 
    f           = 8*abs(x(1)^2 - x(2)) + (1 - x(1))^2;
   
    % GRADIENT AT X
    % Compute the 2nd term
    f_grad      = [-2*(1-x(1)); 0];
    % Add in the 1st term, where we must handle the sign due to the 
    % absolute value
    if x(1)^2 - x(2) >= 0
      f_grad    = f_grad + 8*[ 2*x(1); -1];
    else
      f_grad    = f_grad + 8*[-2*x(1);  1];
    end
end