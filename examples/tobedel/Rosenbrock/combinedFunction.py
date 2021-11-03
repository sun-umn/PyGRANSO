from pygransoStruct import GeneralStruct
# import torch

def eval_obj(X_struct,data_in = None):
    # user defined variable, matirx form. torch tensor
    x1 = X_struct.x1
    x2 = X_struct.x2
    x1.requires_grad_(True)
    x2.requires_grad_(True)
    
    # objective function
    # obtain scalar from the torch tensor
    f = (8 * abs(x1**2 - x2) + (1 - x1)**2)[0,0]
    return f

def combinedFunction(X_struct,data_in = None):
    
    # user defined variable, matirx form. torch tensor
    x1 = X_struct.x1
    x2 = X_struct.x2
    x1.requires_grad_(True)
    x2.requires_grad_(True)
    
    # objective function
    # obtain scalar from the torch tensor
    f = (8 * abs(x1**2 - x2) + (1 - x1)**2)[0,0]

    # inequality constraint, matrix form
    ci = GeneralStruct()
    ci.c1 = (2**0.5)*x1-1  
    ci.c2 = 2*x2-1 

    # equality constraint 
    ce = None

    return [f,ci,ce]