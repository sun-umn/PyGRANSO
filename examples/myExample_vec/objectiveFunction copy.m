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

    % (1) || I - E ||_F
%     f           = (1-x(1))^2  + x(2)^2 + x(3)^2 + (1-x(4))^2; 
%     f_grad      = 2*[x(1)-1;2*x(2);2*x(3);x(4)-1]; 

    % (2) NMF: 0.5||D-UV||_F^2 + lambda/2*||U||_F^2 + lambda/2*||V||_F^2
    f = 0.5 * ( (1-x(1)*x(5)-x(2)*x(7))^2 + (1-x(1)*x(6)-x(2)*x(8))^2 ...
        + (1-x(3)*x(5)-x(4)*x(7))^2 + (1-x(3)*x(6)-x(4)*x(8))^2   );
%     ...
%             + (x(1)^2 + x(2)^2 + x(3)^2 + x(4)^2 + x(5)^2 + x(6)^2 + x(7)^2 + x(8)^2 );
    
    f_grad = [-x(5)*(1-x(1)*x(5)-x(2)*x(7)) - x(6)*(1-x(1)*x(6)-x(2)*x(8))   ;
               -x(7)*(1-x(1)*x(5)-x(2)*x(7)) - x(8)*(1-x(1)*x(6)-x(2)*x(8)) ;
              - x(5)*(1-x(3)*x(5)-x(4)*x(7)) - x(6)*(1-x(3)*x(6)-x(4)*x(8))  ;
              - x(7)*(1-x(3)*x(5)-x(4)*x(7)) - x(8)*(1-x(3)*x(6)-x(4)*x(8))  ;
             -x(1)*(1-x(1)*x(5)-x(2)*x(7)) - x(3)*(1-x(3)*x(5)-x(4)*x(7))   ;
              - x(1)*(1-x(1)*x(6)-x(2)*x(8)) - x(3)*(1-x(3)*x(6)-x(4)*x(8)) ;
              -x(2)*(1-x(1)*x(5)-x(2)*x(7))- x(4)*(1-x(3)*x(5)-x(4)*x(7))   ;
              - x(2)*(1-x(1)*x(6)-x(2)*x(8))- x(4)*(1-x(3)*x(6)-x(4)*x(8)) ];
%               + 2 * x;
end