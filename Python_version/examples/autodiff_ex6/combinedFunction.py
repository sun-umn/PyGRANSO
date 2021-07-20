import numpy as np
from numpy import linalg as LA
from numpy.core.fromnumeric import transpose
import torch
from torch import autograd

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
    u.requires_grad_(True)
    
    # objective function
    # f must be a scalar. not 1 by 1 matrix
    # f = (abs(u))[0][0]
    f = torch.abs(u)
    # print('Gradient function for loss =', f.grad_fn)
    
    # external_grad = torch.tensor([[1.]])
    # f.backward(gradient=external_grad)
    

    # f_grad = f_gradStruct()
    # f.backward()
    # f_grad.u = u.grad

    # #  absolute value
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
    # return [f,ci,ce]