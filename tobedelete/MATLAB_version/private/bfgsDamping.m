function [y,sty,damped] = bfgsDamping(damping,applyH,s,y,sty)
%   bfgsDamping:
%       This function implements Procedure 18.2 from Nocedal and Wright,
%       which ensures that the BFGS update is always well defined. 
%
%   INPUT:
%       damping         [ real number in [0,1] ]
%           constant to determine how frequently and aggressively damping 
%           will be applied.
% 
%       applyH          [ function handle ]
%           Returns H*x, where H is the BFGS inverse Hessian approximation,
%           for single input argument x
%
%       s               [ real finite column vector ]
%           BFGS vector: s = x_{k+1} - x_k = alpha*d
% 
%       y               [ real finite column vector ] 
%           BFGS vector: y = g - gprev
% 
%       sty             [ real finite scalar ]
%           sty = s.'*y
%                   
%   OUTPUT:
%       y               [ real finite column vector ]
%           possibly updated version of y 
%   
%       sty             [ real finite scalar ] 
%           possibly updated version of sty
%
%       damped          [ logical ]
%           true if damping was applied
%        
%
%   If you publish work that uses or refers to GRANSO, please cite the 
%   following paper: 
%
%   [1] Frank E. Curtis, Tim Mitchell, and Michael L. Overton 
%       A BFGS-SQP method for nonsmooth, nonconvex, constrained 
%       optimization and its evaluation using relative minimization 
%       profiles, Optimization Methods and Software, 2016.
%       Available at https://dx.doi.org/10.1080/10556788.2016.1208749
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   bfgsDamping.m introduced in GRANSO Version 1.5.
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
    
    damped      = false;
    Hs          = applyH(s);
    stHs        = s'*Hs;
    
    if sty < damping * stHs
        theta   = ((1 - damping) * stHs) / (stHs - sty);
        y       = theta * y + (1 - theta) * Hs; 
        sty     = theta * sty + (1- theta) * stHs; % s'*y;
        damped  = true;
    end

end