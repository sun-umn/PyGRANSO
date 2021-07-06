import types
import numpy as np
import numpy.linalg as LA
from pygransoStruct import genral_struct

def splitEvalAtX(eval_at_x_fn,x0):
    
    def objective(x):
        [f,g,ci,ci_grad,ce,ce_grad] = eval_at_x_fn(x)
        return [f,g]

    def inequality(varargin):
        c       = ci
        c_grad  = ci_grad
        return [c,c_grad]
    
    def equality(varargin):
        c       = ce
        c_grad  = ce_grad
        return [c,c_grad]

    [f,f_grad,ci,ci_grad,ce,ce_grad] = eval_at_x_fn(x0)
  
    obj_fn = lambda x : objective(x)
    ineq_fn = (lambda varargin: inequality(varargin) ) if (isinstance(ci,np.ndarray)) else (None)
    eq_fn = (lambda varargin: equality(varargin) ) if (isinstance(ce,np.ndarray)) else (None)
    
    
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

def violationsInequality(ci):
    vi = ci
    violated_indx = (ci >= 0)
    vi[~violated_indx] = 0
    return [vi,violated_indx]

def totalViolationMax(v):
    if np.all(v==0):
        v_max = 0
    else:
        v_max = np.max(v)
    return v_max

def totalViolationInequality(ci,ci_grad):
    [vi,indx] = violationsInequality(ci)
    
    #  l_inf penalty term for feasibility measure
    tvi = totalViolationMax(vi)
    
    #  l_1 penalty term for penalty function
    tvi_l1 = np.sum(vi)
    # indx used for select certain cols
    tvi_l1_grad = np.sum(ci_grad[:,indx[:,0]],1)
    return [tvi,tvi_l1,tvi_l1_grad]

def violationsEquality(ce):
    ve = abs(ce)
    violated_indx = (ce >= 0);   #indeed, this is all of them 
    return [ve,violated_indx]

def totalViolationEquality(ce,ce_grad):
    [ve,indx] = violationsEquality(ce)

    # l_inf penalty term for feasibility measure
    tve = totalViolationMax(ve)
    
    # l_1 penalty term for penalty function
    tve_l1 = np.sum(ve)
    tve_l1_grad = np.sum(ce_grad[:,indx[:,0]],1) - sum(ce_grad[:,np.logical_not(indx[:,0])],1)

    return [tve,tve_l1,tve_l1_grad]

def evalInequality(x,fn):
    [ci,ci_grad]                = fn(x)
    [tvi,tvi_l1,tvi_l1_grad]    = totalViolationInequality(ci,ci_grad)
    return

def evalEquality(x,fn):
    [ce,ce_grad]                = fn(x)
    [tve,tve_l1,tve_l1_grad]    = totalViolationEquality(ce,ce_grad)
    return
    

def rescaleConstraint(x,fn,scalings):
    [f,g]   = fn(x)
    f       = np.multiply(f,scalings)
    g       = g @ np.diag(scalings)
    return [f,g]

def setupConstraint( x0, c_fn, eval_fn, inequality_constraint, prescaling_threshold):
                                        
    n = len(x0)
                            
    #  eval_fn is either a function handle for evaluateInequality or
    #  evaluateEquality so we can detect which we have based on its length
    if inequality_constraint:
        viol_fn = lambda ci, ci_grad : totalViolationInequality(ci,ci_grad)
        type_str = "in"
    else:
        viol_fn = lambda ce, ce_grad: totalViolationEquality(ce,ce_grad)
        type_str = ""

    scalings= None; # default if no prescaling is applied
    # isempty function
    if np.all( c_fn==None ):
        eval_fn             = lambda x : None
        # These must have the right dimensions for computations to be 
        # done even if there are no such constraints
        c                   = np.zeros((0,1))
        c_grad              = np.zeros((len(x0),0))
        c_grad_norms        = 0
        tv                  = 0
        tv_l1               = 0
        tv_l1_grad          = 0
        constrained         = False
    elif isinstance(c_fn, types.LambdaType):
        try: 
            [c,c_grad]      = c_fn(x0)
        except Exception as e:
            print(e)
            print("PyGRANSO userSuppliedFunctionsError : failed to evaluate [c,c_grad] = {}eq_fn(x0).".format(type_str))
            
        assertFnOutputs(n,c,c_grad,type_str+"equality constraints") 
        c_grad_norms        = np.sqrt(np.sum(np.square(c_grad),0)) 
        # indices of gradients whose norms are larger than limit
        indx                = c_grad_norms > prescaling_threshold
        if np.any(indx !=0 ):
            scalings        = np.ones(len(c),1)
            # we want to rescale these "too large" functions so that 
            # the norms of their gradients are set to limit at x0
            scalings[indx]  = np.divide(prescaling_threshold, c_grad_norms(indx))
            c_fn            = lambda x: rescaleConstraint(x,c_fn,scalings)
            # rescale already computed constraints and gradients
            c               = np.multiply(c,scalings)
            c_grad          = c_grad @ np.diag(scalings)
        
        [tv,tv_l1,tv_l1_grad]   = viol_fn(c,c_grad)
        # reset eval_fn so that it computes the values and gradients 
        # for both the constraint and the corresponding violation
        eval_fn             = lambda x: eval_fn(x,c_fn)
        constrained         = True
    else:       
        print("PyGRANSO userSuppliedFunctionsError: {}eq_fn must be a function handle of x or empty, that is, None.\n".format(type_str))
    
    return [eval_fn, c, c_grad, tv, tv_l1, tv_l1_grad, c_grad_norms, scalings, constrained]

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
    
    [ eval_ineq_fn,ci,ci_grad,tvi,tvi_l1,tvi_l1_grad,ci_grad_norms,scaling_ci,ineq_constrained] = setupConstraint(x,ineq_fn,lambda x, fn: evalInequality(x,fn),True,prescaling_threshold)

    [eval_eq_fn,ce,ce_grad,tve,tve_l1,tve_l1_grad,ce_grad_norms,scaling_ce,eq_constrained] = setupConstraint(x,eq_fn,lambda x, fn: evalEquality(x,fn),False,prescaling_threshold)

    grad_norms_at_x0 = genral_struct()
    setattr(grad_norms_at_x0,"f",f_grad_norm)
    setattr(grad_norms_at_x0,"ci",ci_grad_norms)
    setattr(grad_norms_at_x0,"ce",ce_grad_norms)

    scalings        = genral_struct()
    if np.any(scaling_f != None):
        setattr(scalings,"f",scaling_f)
    if np.any(scaling_ci != None):
        setattr(scalings,"ci",scaling_ci)
    if np.any(scaling_ce != None):
        setattr(scalings,"ce",scaling_ce)
    prescaled       = len(scalings.__dict__) != 0 

    return [-1,-1]