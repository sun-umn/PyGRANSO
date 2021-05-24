function [f,f_grad,ci,ci_grad,ce,ce_grad] = ...
                            combinedFunction(A,B,C,stability_margin,x)
%   combinedFunction: (examples/ex4)
%       Defines objective and inequality constraint functions, with their
%       respective gradients, for an eigenvalue optimization problem of a 
%       static-output-feedback (SOF) plant:
%       
%           M = A + BXC,
%
%       where A,B,C are all fixed real-valued matrices
%           - A is n by n 
%           - B is n by p
%           - C is m by n
%       and X is a real-valued m by p matrix of the optimization variables.
%
%       The objective function compute the maximum of the imaginary parts
%       of the eigenvalues of M.  In other words, we want to restrict the
%       spectrum of M to be contained in the smallest strip as possible
%       centered on the x-axis (since the spectrum of M is symmetric with
%       respect to the x-axis).
%
%       The (inequality) constraint is that the system must be
%       asymptotically stable, subject to a specified stability margin.  In
%       other words, if the stability margin is 1, the spectral abscissa 
%       must be at most -1.
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
%       [f,f_grad,ci,ci_grad,ce,ce_grad] = ...
%                               combinedFunction(A,B,C,stability_margin,x);
% 
%   INPUT:
%       A                   n by n real-valued matrix
%   
%       B                   n by p real-valued matrix
% 
%       C                   m by n real-valued matrix
%
%       stability_margin    nonnegative real-valued scalar
%
%       x                   optimization variables
%                           real-valued column vector, m*p by 1 
%  
%   OUTPUT:
%       f                   value of the objective function at x
%                           scalar real value
% 
%       f_grad              gradient of the objective function at x.
%                           real-valued column vector, m*p by 1
% 
%       ci                  value of the inequality constraint at x.
%                           scalar real value
% 
%       ci_grad             gradient of the inequality constraint at x.
%                           real-valued column vector, m*p by 1
% 
%       ce                  value of the equality constraint(s) at x.
%                           [], since there aren't any equality constraints
%
%       ce_grad             gradient(s) of the equality constraint(s) at x.
%                           [], since there aren't any equality constraints
%
%
%   For additional nonsmooth constrained optimization problems, using SOF 
%   plants, see:
%
%   PSARNOT: (Pseudo)Spectral Abscissa|Radius Nonsmooth Optimization Test
%   http://www.timmitchell.com/software/PSARNOT
%
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   examples/ex4/combinedFunction.m introduced in GRANSO Version 1.5.
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

    % For both the objective and the constraint, we need to:
    % a) get the dimensions of X 
    % b) reshape x into a matrix X, which is a p by m real-valued matrix
    % c) compute the spectrum and right and left eigenvectors of 
    %    M = A + BXC, where A,B,C are fixed data and X is the matrix of the
    %    optimization variables.
    % As eig(M) is expensive, it would be inefficient to compute it twice,
    % separately for the objective and the constraint.  Using GRANSO's
    % combined format here, we compute it once and then easily share the
    % information for the objective and inequality constraint functions,
    % since they are defined in the same function scope of
    % combinedFunction.
    p           = size(B,2);
    m           = size(C,1);
    X           = reshape(x,p,m);
    M           = A + B*X*C;
    [V,D]       = eig(M);
    d           = diag(D);     
    [VL,D_conj] = eig(M');
    dl          = conj(diag(D_conj));
    
    
    % OBJECTIVE VALUE AT X 
    % Get the max imaginary part, and an eigenvalue associated with it, 
    % since the constraint is to limit eigenvalues to a band centered on 
    % the x-axis 
    [mi,indx]   = max(imag(d));
    indx        = indx(1);
    lambda      = d(indx);
    f           = mi;
    
    % GRADIENT OF THE OBJECTIVE AT X 
    % Get its corresponding right eigenvector
    x           = V(:,indx);
    % Now find the matching left eigenvector for lambda
    [~,indx]    = min(abs(dl - lambda));
    y           = VL(:,indx);
    Bty         = B.'*y;
    Cx          = C*x;
    % Gradient of inner product with respect to A 
    f_grad      = imag((conj(Bty)*Cx.')/(y'*x));
    f_grad      = f_grad(:);
    

    % INEQUALITY CONSTRAINT AT X
    % Compute the spectral abscissa of A from the spectrum and an
    % eigenvalue associated with it
    [ci,indx]   = max(real(d));
    indx        = indx(1); 
    lambda      = d(indx);
    % account for the stability margin in the inequality constraint
    ci          = ci + stability_margin;
    
    % GRADIENT OF THE INEQUALITY CONSTRAINT AT X
    % Get its corresponding right eigenvector
    x           = V(:,indx);
    % Now find the matching left eigenvector for lambda
    [~,indx]    = min(abs(dl - lambda));
    indx        = indx(1);
    y           = VL(:,indx);
    Bty         = B.'*y;
    Cx          = C*x;
    % Gradient of inner product with respect to A 
    ci_grad     = real((conj(Bty)*Cx.')/(y'*x));
    ci_grad     = ci_grad(:);
    
    
    % EQUALITY CONSTRAINT
    % Return [] when (in)equality constraints are not present.
    ce          = [];
    ce_grad     = [];

end