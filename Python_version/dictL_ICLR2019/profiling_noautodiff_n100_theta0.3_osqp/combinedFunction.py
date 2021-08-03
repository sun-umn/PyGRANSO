from pygransoStruct import general_struct
import numpy as np
import numpy.linalg as la

class ci_gradStruct:
    class c1:
        pass

def f_fun(m,Y,q):
    qtY = Y.T @ q
    f = 1/m * la.norm(qtY,  ord = 1)
    return f

def f_gradfun(m,Y,q):
    return 1/m*Y@ np.sign(Y.T@q)

def combinedFunction(X,parameters):
    
    # input variable, matirx form. torch tensor
    q = X.q
    # q.requires_grad_(True)
    
    m = parameters.m
    Y = parameters.Y
    
    # objective function
    f = f_fun(m,Y,q)
    # f = 1/m * np.max(np.sum(np.abs(qtY)))
    
    f_grad = general_struct()
    # f_grad.q = 1/m*Y@ np.sign(Y.T@q)
    f_grad.q = f_gradfun(m,Y,q)

    # inequality constraint, matrix form
    ci = None
    ci_grad = None

    # equality constraint 
    ce = general_struct()
    ce.c1 = q.T @ q - 1

    ce_grad = ci_gradStruct()
    ce_grad.c1.q = 2*q

    return [f,f_grad,ci,ci_grad,ce,ce_grad]


