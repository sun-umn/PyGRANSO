def linesearchWeakWolfe( x0, f0, grad0, d, obj_fn,
                        c1, c2, fvalquit, eval_limit, step_tol):
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
    
    print("TODO: linesearchWeakWolfe")
    # return[  alpha, xalpha, falpha, gradalpha, fail, beta, gradbeta, n_evals]  
    return[  -1,-1,-1,-1,-1,-1,-1,-1]                                        