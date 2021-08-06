from pygransoStruct import general_struct
import torch

def combinedFunction(X_struct,parameters = None):
    
    # user defined variable, matirx form. torch tensor
    x = X_struct.x
    x.requires_grad_(True)
    
    # obtain parameters from runExample.py
    b = parameters.b
    F = parameters.F
    eta = parameters.eta
    
    # objective function
    f = (x-b).t() @ (x-b)
    f += eta * torch.norm( F@x, p = 1)

    # inequality constraint, matrix form
    ci = None

    # equality constraint 
    ce = None

    return [f,ci,ce]