import numpy as np
from numpy import linalg as LA
from numpy.core.fromnumeric import transpose
import torch

class f_gradStruct:
    pass

class ciStruct:
    pass

class ci_gradStruct:
    class c1:
        pass
    class c2:
        pass

def combinedFunction(X):
    
    # input variable, matirx form
    u = X.u
    v = X.v
    u.requires_grad_(True)
    v.requires_grad_(True)
    

    # objective function
    # f must be a scalar. not 1 by 1 matrix
    f = (torch.abs(u) + v**2 )[0][0]

    # f_grad = f_gradStruct()
    # f_grad.v = 2*v
    # #  Compute the 2nd term
    # if u >= 0:
    #   f_grad.u    = 1
    # else:
    #   f_grad.u    = -1

    
    # inequality constraint 
    ci = None
    # ci_grad = None
    
    # equality constraint 
    ce = None
    # ce_grad = None

    return [f,ci,ce]