from pygransoStruct import GeneralStruct
import torch

def eval_obj(X_struct,data_in = None):
    # user defined variable, matirx form. torch tensor
    x = X_struct.x
    x.requires_grad_(True)
    
    # obtain data_in from runExample.py
    b = data_in.b
    F = data_in.F
    eta = data_in.eta
    
    # objective function
    f = (x-b).t() @ (x-b)  + eta * torch.norm( F@x, p = 1)
    return f

def combinedFunction(X_struct,data_in = None):
    
    # user defined variable, matirx form. torch tensor
    x = X_struct.x
    x.requires_grad_(True)
    
    # obtain data_in from runExample.py
    b = data_in.b
    F = data_in.F
    eta = data_in.eta
    
    # objective function
    f = (x-b).t() @ (x-b)  + eta * torch.norm( F@x, p = 1)

    # inequality constraint, matrix form
    ci = None

    # equality constraint 
    ce = None

    return [f,ci,ce]