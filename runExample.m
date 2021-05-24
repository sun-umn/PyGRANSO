function soln = runExample()
%   runExample: (examples/ex1)
%       Run GRANSO on a 2-variable nonsmooth Rosenbrock objective function,
%       subject to simple bound constraints, with GRANSO's default
%       parameters.
%    
%       Read this source code.
%   
%       This tutorial example shows:
%
%           - how to call GRANSO using objective and constraint functions
%             defined in .m files 
%       
%           - how to set GRANSO's inputs when there aren't any 
%             equality constraint functions (which also applies when there
%             aren't any inequality constraints)
%
%   USAGE:
%       soln = runExample();
% 
%   INPUT: [none]
%   
%   OUTPUT:
%       soln        GRANSO's output struct
%
%   See also objectiveFunction, inequalityConstraint. 
% 
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   examples/ex1/runExample.m introduced in GRANSO Version 1.5.
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
    
    % Call GRANSO using its "separate" format, where objective and
    % constraint functions are computed by separate .m files.
    % By default, GRANSO will generate an initial point via randn() .
    
    nvar    = 2;    % the number of optimization variables is 2
    eq_fn   = [];   % use [] when (in)equality constraints aren't present
    soln    = granso(nvar,@objectiveFunction,@inequalityConstraint,eq_fn);
    
    % Alternatively, without the eq_fn variable:
    % soln    = granso(nvar,@objectiveFunction,@inequalityConstraint,[]);
    
end