function [  alpha,          ...
            xalpha,         ...
            falpha,         ...
            gradalpha,      ...
            fail,           ...
            beta,           ...
            gradbeta,       ...
            n_evals] = linesearchWeakWolfe( x0, f0, grad0, d,           ...
                                            obj_fn,                     ...
                                            c1, c2,                     ...
                                            fvalquit, eval_limit, step_tol)
% linesearchWeakWolfe:
%   Line search enforcing weak Wolfe conditions, suitable for minimizing 
%   both smooth and nonsmooth functions.  This routine is a slightly 
%   modified version of linesch_ww.m from HANSO 2.1, to faciliate a few 
%   different input and output arguments but the method itself remains 
%   unchanged.  The function name has been changed so that they are not
%   mistakenly used in lieu of one another.  
%   NOTE: the values assigned to output argument "fail" have been changed 
%         so that all error cases are assigned positive codes.
%       
% call:  
% [alpha, xalpha, falpha, gradalpha, fail, beta, gradbeta, n_evals] = ...
%   linesearchWeakWolfe(    x0, f0, grad0, d, obj_fn,   ...
%                           c1, c2, fvalquit, eval_limitprtlevel);
%  Input
%   x0:             intial point
%   f0:             function value at x0
%   grad0:          gradient at x0
%   d:              search direction  
%   obj_fn:         a function handle for evaluating the objective function
%                   (the penalty function for constrained problems) and its
%                   gradient at some vector x, along with a logical
%                   indicating whether this x is considered sufficiently 
%                   close to the feasible region.
%                   NOTE:   for unconstrained problems, this logical (set 
%                           as true) must still be returned 
%                   e.g. [f,g,is_feasible] = obj_fn(x)         
%   c1: Wolfe parameter for the sufficient decrease condition 
%          f(x0 + t d) ** < ** f0 + c1*t*grad0'*d     (DEFAULT 0)
%   c2: Wolfe parameter for the WEAK condition on directional derivative
%          (grad f)(x0 + t d)'*d ** > ** c2*grad0'*d  (DEFAULT 0.5)
%        where 0 <= c1 <= c2 <= 1.
%        For usual convergence theory for smooth functions, normally one
%        requires 0 < c1 < c2 < 1, but c1=0 is fine in practice.
%        May want c1 = c2 = 0 for some nonsmooth optimization 
%        algorithms such as Shor or bundle, but not BFGS.
%        Setting c2=0 may interfere with superlinear convergence of
%        BFGS in smooth case.
%   fvalquit: quit immediately if f drops below this value, regardless
%        of the Wolfe conditions (default -inf)
%   eval_limit: line search quits after eval_limit calls to obj_fn
%   step_tol: determines how small the step is allowed to get
%
%  Output:
%   alpha:   steplength satisfying weak Wolfe conditions if one was found,
%             otherwise left end point of interval bracketing such a point
%             (possibly 0)
%   xalpha:  x0 + alpha*d
%   is_feasible: if xalpha is considered feasible or not
%   falpha:  f(x0 + alpha d)
%   gradalpha:(grad f)(x0 + alpha d)  
%   fail:    0 if both Wolfe conditions satisfied, or falpha < fvalquit
%            1 if one or both Wolfe conditions not satisfied but an
%               interval was found bracketing a point where both satisfied
%            2 if no such interval was found, function may be unbounded below
%   beta:    same as alpha if it satisfies weak Wolfe conditions,
%             otherwise right end point of interval bracketing such a point
%             (inf if no such finite interval found)
%   gradbeta: (grad f)(x0 + beta d) (this is important for bundle methods)
%             (vector of nans if beta is inf)        
%   n_evals:  number of incurred function evaluations
%
% The weak Wolfe line search is far less complicated that the standard 
% strong Wolfe line search that is discussed in many texts. It appears
% to have no disadvantages compared to strong Wolfe when used with
% Newton or BFGS methods on smooth functions, and it is essential for the 
% application of BFGS or bundle to nonsmooth functions as done in HANSO.
% However, it is NOT recommended for use with conjugate gradient methods,
% which require a strong Wolfe line search for convergence guarantees.
% Weak Wolfe requires two conditions to be satisfied: sufficient decrease
% in the objective, and sufficient increase in the directional derivative
% (not reduction in its absolute value, as required by strong Wolfe).
%
% There are some subtleties for nonsmooth functions.  In the typical case
% that the directional derivative changes sign somewhere along d, it is
% no problem to satisfy the 2nd condition, but descent may not be possible
% if the change of sign takes place even when the step is tiny. In this
% case it is important to return the gradient corresponding to the positive 
% directional derivative even though descent was not obtained. On the other 
% hand, for some nonsmooth functions the function decrease is steady
% along the line until at some point it jumps to infinity, because an
% implicit constraint is violated.  In this case, the first condition is
% satisfied but the second is not. All cases are covered by returning
% the end points of an interval [alpha, beta] and returning the function 
% value at alpha, but the gradients at both alpha and beta. 
%
% The assertion that [alpha,beta] brackets a point satisfying the
% weak Wolfe conditions depends on an assumption that the function 
% f(x + td) is a continuous and piecewise continuously differentiable 
% function of t, and that in the unlikely event that f is evaluated at
% a point of discontinuity of the derivative, g'*d, where g is the 
% computed gradient, is either the left or right derivative at the point
% of discontinuity, or something in between these two values.
%
% For functions that are known to be nonsmooth, setting the second Wolfe
% parameter to zero makes sense, especially for a bundle method, and for
% the Shor R-algorithm, for which it is essential.  However, it's not
% a good idea for BFGS, as for smooth functions this may prevent superlinear 
% convergence, and it can even make trouble for BFGS on, e.g., 
% f(x) = x_1^2 + eps |x_2|, when eps is small.
%
% Line search quits immediately if f drops below fvalquit and the iterate
% is considered to be feasible.
% 
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   linesearchWeakWolfe.m introduced in GRANSO Version 1.0.
%
% =========================================================================
% |  linesearchWeakWolfe.m                                                |
% |  Copyright (C) 2016 James Burke, Adrian Lewis, Tim Mitchell, and      |
% |  Michael Overton                                                      |
% |                                                                       |
% |  This routine is a modified version of the linesch_ww.m routine from  |
% |  the HANSO software package, which is licensed under the GPL v3.  As  |
% |  such, this single routine is also licensed under the GPL v3.         |
% |  However, note that this is an exceptional case; GRANSO and most of   |
% |  its subroutines are licensed under the AGPL v3.                      |
% |                                                                       | 
% |  This routine (this single file) is free software: you can            |
% |  redistribute it and/or modify it under the terms of the GNU General  |
% |  Public License as published by the Free Software Foundation, either  |
% |  version 3 of the License, or (at your option) any later version.     |
% |                                                                       |
% |  This routine is distributed in the hope that it will be useful, but  |
% |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
% |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    |
% |  General Public License for more details.                             |
% |                                                                       |
% |  You should have received a copy of the GNU General Public License    |
% |  along with this program.  If not, see                                |
% |  <http://www.gnu.org/licenses/gpl.html>.                              |
% =========================================================================
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

if nargin < 6  % check if the optional Wolfe parameters were passed
    c1 = 0; % not conventional, but seems OK.  See note at top.
end
if nargin < 7
    c2 = 0.5; % see note at top
end
if nargin < 8
    fvalquit = -inf;
end
if nargin < 9
    eval_limit = inf;
end
if nargin < 10
    step_tol = 1e-12;
end
alpha = 0;  % lower bound on steplength conditions
xalpha = x0;
falpha = f0;
gradalpha = grad0; % need to pass grad0, not grad0'*d, in case line search fails
beta = inf;  % upper bound on steplength satisfying weak Wolfe conditions
gradbeta = nan*ones(size(x0));
g0 = grad0'*d; 
dnorm = norm(d);
t = 1;  % important to try steplength one first
n_evals = 0;
nexpand = 0;
% the following limit is rather arbitrary
% don't use HANSO's nexpandmax, which could much larger, since BFGS-SQP 
% will automatically reattempt the line search with a lower penalty 
% parameter if it terminates with the "f may be unbounded below" case.
nexpandmax = max(10, round(log2(1e5/dnorm))); % allows more if ||d|| small

while (beta - alpha) > (norm(x0 + alpha*d)/dnorm)*step_tol && n_evals < eval_limit
    x = x0 + t*d;
    [f,grad,is_feasible] = obj_fn(x);
    n_evals = n_evals + 1;
    if is_feasible && ~isnan(f) && f <= fvalquit && ~isinf(f) 
        fail = 0;
        alpha = t; % normally beta is inf
        xalpha = x;
        falpha = f;
        gradalpha = grad;
        return
    end
    gtd = grad'*d;
    % the first condition must be checked first. NOTE THE >=.
    if f >= f0 + c1*t*g0 || isnan(f) % first condition violated, gone too far
        beta = t;
        gradbeta = grad; % discard f
    % now the second condition.  NOTE THE <=
    elseif gtd <= c2*g0 || isnan(gtd) % second condition violated, not gone far enough
        alpha = t;
        xalpha = x;
        falpha = f;
        gradalpha = grad;
    else   % quit, both conditions are satisfied
        fail = 0;
        alpha = t;
        xalpha = x;
        falpha = f;
        gradalpha = grad;
        beta = t;
        gradbeta = grad;
        return
    end
    % setup next function evaluation
    if beta < inf
        t = (alpha + beta)/2; % bisection
    elseif nexpand < nexpandmax
        nexpand = nexpand + 1;
        t = 2*alpha;  % still in expansion mode
    else
        break % Reached the maximum number of expansions
    end
end % loop
% Wolfe conditions not satisfied: there are two cases
if beta == inf % minimizer never bracketed
    fail = 2;
else % point satisfying Wolfe conditions was bracketed
    fail = 1;
end
end