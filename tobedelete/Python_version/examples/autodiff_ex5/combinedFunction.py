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
    a = 20

    # objective function
    # f must be a scalar. not 1 by 1 matrix
    f = (a*torch.abs(u) + torch.sum(v))[0][0]

    # f_grad = f_gradStruct()
    # #  Compute the 2nd term
    # f_grad.v = np.ones((len(v),1))
    # #  Add in the 1st term, where we must handle the sign due to the 
    # #  absolute value
    # if u >= 0:
    #   f_grad.u    = a
    # else:
    #   f_grad.u    = -a

    
    # inequality constraint 
    ci = None
    # ci_grad = None
    
    # equality constraint 
    ce = None
    # ce_grad = None

    return [f,ci,ce]