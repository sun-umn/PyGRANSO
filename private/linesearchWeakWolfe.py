import numpy as np
import numpy.linalg as LA
import math
from numpy import conjugate as conj
from dbg_print import dbg_print_1

def linesearchWeakWolfe( x0, f0, grad0, d, obj_fn, c1 = 0, c2 = 0.5, fvalquit = -np.inf, eval_limit = np.inf, step_tol = 1e-12):
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
    """

    dbg_print_1("start rescaling line search direction:")
    dbg_print_1("norm of d = {}".format(LA.norm(d)))
    d_rescale = d.copy()
    d_rescale =  d_rescale / LA.norm(d_rescale)
    # d_rescale = 5 * d_rescale

    dbg_print_1("end rescaling line search direction.")


    alpha = 0  # lower bound on steplength conditions
    xalpha = x0.copy()
    falpha = f0
    gradalpha = grad0.copy() # need to pass grad0, not grad0'*d, in case line search fails
    beta = np.inf  # upper bound on steplength satisfying weak Wolfe conditions
    gradbeta = np.empty(x0.shape)
    gradbeta[:] = np.nan
    # g0 = conj(grad0.T) @ d 
    # dnorm = LA.norm(d)
    g0 = conj(grad0.T) @ d_rescale 
    dnorm = LA.norm(d_rescale)
    t = 1  # important to try steplength one first
    n_evals = 0
    nexpand = 0
    # the following limit is rather arbitrary
    # don't use HANSO's nexpandmax, which could much larger, since BFGS-SQP 
    # will automatically reattempt the line search with a lower penalty 
    # parameter if it terminates with the "f may be unbounded below" case.
    nexpandmax = max(10, round(math.log2(1e5/dnorm)))  # allows more if ||d|| small

    test_flag = 0

    # while (beta - alpha) > (LA.norm(x0 + alpha*d)/dnorm)*step_tol and n_evals < eval_limit:
    #     x = x0 + t*d
    while (beta - alpha) > (LA.norm(x0 + alpha*d_rescale)/dnorm)*step_tol and n_evals < eval_limit:
        x = x0 + t*d_rescale
        [f,grad,is_feasible] = obj_fn(x)
        n_evals = n_evals + 1
        if is_feasible and not np.isnan(f) and f <= fvalquit and not np.isinf(f): 
            fail = 0
            alpha = t  # normally beta is inf
            xalpha = x.copy()
            falpha = f
            gradalpha = grad.copy()
            return [alpha, xalpha, falpha, gradalpha, fail, beta, gradbeta, n_evals] 
        
        # gtd = conj(grad.T) @ d
        gtd = conj(grad.T) @ d_rescale
        #  the first condition must be checked first. NOTE THE >=.
        if f >= f0 + c1*t*g0 or np.isnan(f): # first condition violated, gone too far
            beta = t
            gradbeta = grad.copy() # discard f
            test_flag = 1
        #  now the second condition.  NOTE THE <=
        elif gtd <= c2*g0 or np.isnan(gtd): # second condition violated, not gone far enough
            alpha = t
            xalpha = x.copy()
            falpha = f
            gradalpha = grad.copy()
            test_flag = 2
        else:   # quit, both conditions are satisfied
            fail = 0
            alpha = t
            xalpha = x.copy()
            falpha = f
            gradalpha = grad.copy()
            beta = t
            gradbeta = grad.copy()
            return [alpha, xalpha, falpha, gradalpha, fail, beta, gradbeta, n_evals] 
        
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
        dbg_print_1("wolfe condition %d fails"%test_flag)
        fail = 1
    

    return [alpha, xalpha, falpha, gradalpha, fail, beta, gradbeta, n_evals]                               