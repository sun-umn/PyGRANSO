import numpy as np
import math
import torch

def linesearchWeakWolfe( x0, f0, grad0, d, f_eval_fn, obj_fn, c1 = 0, c2 = 0.5, fvalquit = -np.inf, eval_limit = np.inf, step_tol = 1e-12, init_step_size = 1, linesearch_maxit = np.inf, is_backtrack_linesearch = False, torch_device = torch.device('cpu')):
    """
    linesearchWeakWolfe:
        Line search enforcing weak Wolfe conditions, suitable for minimizing 
        both smooth and nonsmooth functions.  This routine is a slightly 
        modified version of linesch_ww.m from HANSO 2.1, to faciliate a few 
        different input and output arguments but the method itself remains 
        unchanged.  The function name has been changed so that they are not
        mistakenly used in lieu of one another.  

        NOTE: the values assigned to output argument "fail" have been changed 
                so that all error cases are assigned positive codes.
            
        call:  
        [alpha, xalpha, falpha, gradalpha, fail, beta, gradbeta, n_evals] = ...
        linesearchWeakWolfe( x0, f0, grad0, d, f_eval_fn, obj_fn, c1 = 0, 
                            c2 = 0.5, fvalquit = -np.inf, eval_limit = np.inf, 
                            step_tol = 1e-12, init_step_size = 1, linesearch_maxit = np.inf, 
                            is_backtrack_linesearch = False, torch_device = torch.device('cpu'))

        Input
            x0:             intial point
            f0:             function value at x0
            grad0:          gradient at x0
            d:              search direction  
            obj_fn:         a function handle for evaluating the objective function
                            (the penalty function for constrained problems) and its
                            gradient at some vector x, along with a logical
                            indicating whether this x is considered sufficiently 
                            close to the feasible region.
                            NOTE:   for unconstrained problems, this logical (set 
                                    as true) must still be returned 
                            e.g. [f,g,is_feasible] = obj_fn(x)         
            c1: Wolfe parameter for the sufficient decrease condition 
                    f(x0 + t d) ** < ** f0 + c1*t*grad0'*d     (DEFAULT 0)
            c2: Wolfe parameter for the WEAK condition on directional derivative
                    (grad f)(x0 + t d)'*d ** > ** c2*grad0'*d  (DEFAULT 0.5)
                where 0 <= c1 <= c2 <= 1.
                For usual convergence theory for smooth functions, normally one
                requires 0 < c1 < c2 < 1, but c1=0 is fine in practice.
                May want c1 = c2 = 0 for some nonsmooth optimization 
                algorithms such as Shor or bundle, but not BFGS.
                Setting c2=0 may interfere with superlinear convergence of
                BFGS in smooth case.
            fvalquit: quit immediately if f drops below this value, regardless
                of the Wolfe conditions (default -inf)
            eval_limit: line search quits after eval_limit calls to obj_fn
            step_tol: determines how small the step is allowed to get
        
        Output:
            alpha:   steplength satisfying weak Wolfe conditions if one was found,
                        otherwise left end point of interval bracketing such a point
                        (possibly 0)
            xalpha:  x0 + alpha*d
            is_feasible: if xalpha is considered feasible or not
            falpha:  f(x0 + alpha d)
            gradalpha:(grad f)(x0 + alpha d)  
            fail:    0 if both Wolfe conditions satisfied, or falpha < fvalquit
                    1 if one or both Wolfe conditions not satisfied but an
                        interval was found bracketing a point where both satisfied
                    2 if no such interval was found, function may be unbounded below
            beta:    same as alpha if it satisfies weak Wolfe conditions,
                        otherwise right end point of interval bracketing such a point
                        (inf if no such finite interval found)
            gradbeta: (grad f)(x0 + beta d) (this is important for bundle methods)
                        (vector of nans if beta is inf)        
            n_evals:  number of incurred function evaluations
        
        The weak Wolfe line search is far less complicated that the standard 
        strong Wolfe line search that is discussed in many texts. It appears
        to have no disadvantages compared to strong Wolfe when used with
        Newton or BFGS methods on smooth functions, and it is essential for the 
        application of BFGS or bundle to nonsmooth functions as done in HANSO.
        However, it is NOT recommended for use with conjugate gradient methods,
        which require a strong Wolfe line search for convergence guarantees.
        Weak Wolfe requires two conditions to be satisfied: sufficient decrease
        in the objective, and sufficient increase in the directional derivative
        (not reduction in its absolute value, as required by strong Wolfe).
        
        There are some subtleties for nonsmooth functions.  In the typical case
        that the directional derivative changes sign somewhere along d, it is
        no problem to satisfy the 2nd condition, but descent may not be possible
        if the change of sign takes place even when the step is tiny. In this
        case it is important to return the gradient corresponding to the positive 
        directional derivative even though descent was not obtained. On the other 
        hand, for some nonsmooth functions the function decrease is steady
        along the line until at some point it jumps to infinity, because an
        implicit constraint is violated.  In this case, the first condition is
        satisfied but the second is not. All cases are covered by returning
        the end points of an interval [alpha, beta] and returning the function 
        value at alpha, but the gradients at both alpha and beta. 
        
        The assertion that [alpha,beta] brackets a point satisfying the
        weak Wolfe conditions depends on an assumption that the function 
        f(x + td) is a continuous and piecewise continuously differentiable 
        function of t, and that in the unlikely event that f is evaluated at
        a point of discontinuity of the derivative, g'*d, where g is the 
        computed gradient, is either the left or right derivative at the point
        of discontinuity, or something in between these two values.
        
        For functions that are known to be nonsmooth, setting the second Wolfe
        parameter to zero makes sense, especially for a bundle method, and for
        the Shor R-algorithm, for which it is essential.  However, it's not
        a good idea for BFGS, as for smooth functions this may prevent superlinear 
        convergence, and it can even make trouble for BFGS on, e.g., 
        f(x) = x_1^2 + eps |x_2|, when eps is small.
        
        Line search quits immediately if f drops below fvalquit and the iterate
        is considered to be feasible.

        If you publish work that uses or refers to NCVX, please cite both
        NCVX and GRANSO paper:

        [1] Buyun Liang, and Ju Sun. 
            NCVX: A User-Friendly and Scalable Package for Nonconvex 
            Optimization in Machine Learning. arXiv preprint arXiv:2111.13984 (2021).
            Available at https://arxiv.org/abs/2111.13984

        [2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton 
            A BFGS-SQP method for nonsmooth, nonconvex, constrained 
            optimization and its evaluation using relative minimization 
            profiles, Optimization Methods and Software, 32(1):148-181, 2017.
            Available at https://dx.doi.org/10.1080/10556788.2016.1208749
            
        Change Log:
            linesearchWeakWolfe.m introduced in GRANSO Version 1.0.
            
            Buyun Dec 20, 2021 (NCVX Version 1.0.0):
                linesearchWeakWolfe.py is translated from linesearchWeakWolfe.m in GRANSO Version 1.6.4. 

                backtracking line search and related option added.

        For comments/bug reports, please visit the NCVX webpage:
        https://github.com/sun-umn/NCVX
            
        NCVX Version 1.0.0, 2021, see AGPL license info below.

        =========================================================================
        |  linesearchWeakWolfe.m                                                |
        |  Copyright (C) 2016 James Burke, Adrian Lewis, Tim Mitchell, and      |
        |  Michael Overton                                                      |
        |                                                                       |
        |  This routine is a modified version of the linesch_ww.m routine from  |
        |  the HANSO software package, which is licensed under the GPL v3.  As  |
        |  such, this single routine is also licensed under the GPL v3.         |
        |  However, note that this is an exceptional case; GRANSO and most of   |
        |  its subroutines are licensed under the AGPL v3.                      |
        |                                                                       | 
        |  This routine (this single file) is free software: you can            |
        |  redistribute it and/or modify it under the terms of the GNU General  |
        |  Public License as published by the Free Software Foundation, either  |
        |  version 3 of the License, or (at your option) any later version.     |
        |                                                                       |
        |  This routine is distributed in the hope that it will be useful, but  |
        |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
        |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    |
        |  General Public License for more details.                             |
        |                                                                       |
        |  You should have received a copy of the GNU General Public License    |
        |  along with this program.  If not, see                                |
        |  <http://www.gnu.org/licenses/gpl.html>.                              |
        =========================================================================

        =========================================================================
        |  GRANSO: GRadient-based Algorithm for Non-Smooth Optimization         |
        |  Copyright (C) 2016 Tim Mitchell                                      |
        |                                                                       |
        |  This file is translated from GRANSO.                                 |
        |                                                                       |
        |  GRANSO is free software: you can redistribute it and/or modify       |
        |  it under the terms of the GNU Affero General Public License as       |
        |  published by the Free Software Foundation, either version 3 of       |
        |  the License, or (at your option) any later version.                  |
        |                                                                       |
        |  GRANSO is distributed in the hope that it will be useful,            |
        |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
        |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
        |  GNU Affero General Public License for more details.                  |
        |                                                                       |
        |  You should have received a copy of the GNU Affero General Public     |
        |  License along with this program.  If not, see                        |
        |  <http://www.gnu.org/licenses/agpl.html>.                             |
        =========================================================================

        =========================================================================
        |  NCVX (NonConVeX): A User-Friendly and Scalable Package for           |
        |  Nonconvex Optimization in Machine Learning.                          |
        |                                                                       |
        |  Copyright (C) 2021 Buyun Liang                                       |
        |                                                                       |
        |  This file is part of NCVX.                                           |
        |                                                                       |
        |  NCVX is free software: you can redistribute it and/or modify         |
        |  it under the terms of the GNU Affero General Public License as       |
        |  published by the Free Software Foundation, either version 3 of       |
        |  the License, or (at your option) any later version.                  |
        |                                                                       |
        |  GRANSO is distributed in the hope that it will be useful,            |
        |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
        |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
        |  GNU Affero General Public License for more details.                  |
        |                                                                       |
        |  You should have received a copy of the GNU Affero General Public     |
        |  License along with this program.  If not, see                        |
        |  <http://www.gnu.org/licenses/agpl.html>.                             |
        =========================================================================

    """

    alpha = 0  # lower bound on steplength conditions
    xalpha = x0.detach().clone()
    falpha = f0
    gradalpha = grad0.detach().clone() # need to pass grad0, not grad0'*d, in case line search fails
    beta = float('inf')  # upper bound on steplength satisfying weak Wolfe conditions

    g0 = (torch.conj(grad0.t()) @ d).item() 
    dnorm = torch.norm(d).item()
    t = init_step_size
    n_evals = 0
    nexpand = 0
    maxit = min(eval_limit,linesearch_maxit)

    # the following limit is rather arbitrary
    # don't use HANSO's nexpandmax, which could much larger, since BFGS-SQP 
    # will automatically reattempt the line search with a lower penalty 
    # parameter if it terminates with the "f may be unbounded below" case.
    nexpandmax = max(10, round(math.log2(1e5/dnorm)))  # allows more if ||d|| small

    while (beta - alpha) > (torch.norm(x0 + alpha*d).item()/dnorm)*step_tol and n_evals < maxit:
        x = x0 + t*d

        if is_backtrack_linesearch:
            [f,is_feasible] = f_eval_fn(x)
            # random setting, avoid error in 2nd wolfe condition
            gtd = 2*(torch.conj(grad0.t()) @ d)
        else:
            [f,grad,is_feasible] = obj_fn(x)
            gtd = torch.conj(grad.t()) @ d

        if torch.is_tensor(f):
            f = f.item()
        n_evals = n_evals + 1
        if is_feasible and not np.isnan(f) and f <= fvalquit and not np.isinf(f): 
            fail = 0
            alpha = t  # normally beta is inf
            xalpha = x.detach().clone()
            [f,grad,is_feasible] = obj_fn(x)
            falpha = f
            gradalpha = grad.detach().clone()
            return [alpha, xalpha, falpha, gradalpha, fail] 
        
        #  the first condition must be checked first. NOTE THE >=.
        if f >= f0 + c1*t*g0 or np.isnan(f): # first condition violated, gone too far
            beta = t

        elif not is_backtrack_linesearch and gtd <= c2*g0 or torch.isnan(gtd): # second condition violated, not gone far enough
                alpha = t
                xalpha = x.detach().clone()
                falpha = f
                gradalpha = grad.detach().clone()

        else:   # quit, both conditions are satisfied
            fail = 0
            alpha = t
            xalpha = x.detach().clone()
            if is_backtrack_linesearch:
                [f,grad,is_feasible] = obj_fn(x)
            falpha = f
            gradalpha = grad.detach().clone()
            beta = t
            return [alpha, xalpha, falpha, gradalpha, fail] 
        
        #  setup next function evaluation
        if beta < np.inf:
            t = (alpha + beta)/2 # bisection
        elif nexpand < nexpandmax:
            nexpand = nexpand + 1
            t = 2*alpha  # still in expansion mode
        else:
            break # Reached the maximum number of expansions

    # end loop
    # Wolfe conditions not satisfied: there are two cases
    if beta == np.inf: # minimizer never bracketed
        fail = 2
    else: # point satisfying Wolfe conditions was bracketed
        fail = 1
    
    #####################################################################
    if is_backtrack_linesearch:
        alpha = t
        xalpha = x.detach().clone()
        [f,grad,is_feasible] = obj_fn(x)
        falpha = f
        gradalpha = grad.detach().clone()
        beta = t

    return [alpha, xalpha, falpha, gradalpha, fail]                              