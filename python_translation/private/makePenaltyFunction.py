def makePenaltyFunction(params,obj_fn,varargin=None):
    """
    makePenaltyFunction: 
        creates an object representing the penalty function for 
            min obj_fn 
            subject to ineq_fn <= 0
                            eq_fn == 0
        where the penalty function is specified with initial penalty 
        parameter value mu and is applied to the objective function.
        Roughly, this means:
            mu*obj_fn   + sum of active inequality constraints 
                        + sum of absolute value of eq. constraints
    """


    return [-1,-1]