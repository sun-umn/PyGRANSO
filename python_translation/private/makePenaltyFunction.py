import types
import numpy as np
import numpy.linalg as LA

def splitEvalAtX(eval_at_x_fn,x0):
    
    def objective(x):
        [f,g,ci,ci_grad,ce,ce_grad] = eval_at_x_fn(x)
        return [f,g]

    def inequality():
        c       = ci
        c_grad  = ci_grad
        return [c,c_grad]
    
    def equality():
        c       = ce
        c_grad  = ce_grad
        return [c,c_grad]

    [f,f_grad,ci,ci_grad,ce,ce_grad] = eval_at_x_fn(x0)
  
    obj_fn = lambda x : objective(x)
    ineq_fn = (lambda : inequality() ) if (isinstance(ci,np.ndarray)) else None
    eq_fn = (lambda : equality() ) if (isinstance(ce,np.ndarray)) else None
    
    
    return [f,f_grad,obj_fn,ineq_fn,eq_fn] 

def assertFn(cond,arg_name,fn_name,msg):
    assert cond,("PyGRANSO userSuppliedFunctionsError: The {} at x0 returned by the {} function should {}!".format(arg_name,fn_name,msg)  )                                 

def assertFnOutputs(n,f,g,fn_name):
    if fn_name == "objective":
        arg1 = "function value"
        arg2 = "gradient"
        assertFn(np.isscalar(f),arg1,fn_name,'be a scalar')
        [r,c] = g.shape
        assertFn(c == 1,arg2,fn_name,'be a column vector')
    else:
        arg1 = "function value(s)"
        arg2 = "gradient(s)"
        [nf,c] = f.shape
        assertFn(nf >= 1 and c == 1,arg1,fn_name,'be a column vector')
        [r,ng] = g.shape
        assertFn(nf == ng,'number of gradients',fn_name,'should match the number of constraint function values' )

    assertFn(r == n,arg2,fn_name,'have dimension matching the number of variables')

    assertFn(np.isreal(f.all()),arg1,fn_name,'should be real valued')
    assertFn(np.isreal(g.all()),arg2,fn_name,'should be real valued')
    assertFn(np.isfinite(f.all()),arg1,fn_name,'should be finite valued')
    assertFn(np.isfinite(g.all()),arg2,fn_name,'should be finite valued')
    return

def rescaleObjective(x,fn,scaling):
    [f,g]   = fn(x)
    f       = f*scaling
    g       = g*scaling
    return [f,g]

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

    # assert (isinstance(obj_fn,types.LambdaType), 'PyGRANSO userSuppliedFunctionsError: obj_fn must be a lambda function of x.' )
    
    # local storage for function and gradients and the current x
    
    x = params.x0
    n = len(x)

    # objective and its gradient
  
    # originally: if nargin < 4. Currently we only allow combinefns
    try: 
        [f,f_grad,obj_fn,ineq_fn,eq_fn] = splitEvalAtX(obj_fn,x)
    except Exception as e:
        print(e)         
        print("\n error handler TODO: failed to evaluate [f,grad,ci,ci_grad,ce,ce_grad] = obj_fn(x0). \n")
    
    assertFnOutputs(n,f,f_grad,'objective')

    prescaling_threshold = params.prescaling_threshold
    # checking scaling of objective and rescale if necessary
    f_grad_norm = LA.norm(f_grad)
    if f_grad_norm > prescaling_threshold:
        scaling_f = prescaling_threshold / f_grad_norm
        obj_fn = lambda x : rescaleObjective(x,obj_fn,scaling_f)
        f = f * scaling_f
        f_grad = f_grad * scaling_f
    else:
        scaling_f = None

    # setup inequality and equality constraints, violations, and scalings
    
    # [ eval_ineq_fn,ci,ci_grad,tvi,tvi_l1,tvi_l1_grad,ci_grad_norms,scaling_ci,ineq_constrained] = setupConstraint(x,ineq_fn,@evalInequality,true,prescaling_threshold)


    return [-1,-1]