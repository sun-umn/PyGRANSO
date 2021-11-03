from pygransoStruct import GeneralStruct
import torch

def eval_obj(X_struct,data_in = None):
# user defined variable, matirx form. torch tensor
    q = X_struct.q
    q.requires_grad_(True)
    
    # obtain data_in from runExample.py
    m = data_in.m
    Y = data_in.Y
    
    # objective function
    qtY = q.t() @ Y
    f = 1/m * torch.norm(qtY, p = 1)
    return f

def combinedFunction(X_struct, data_in = None):
    
    # user defined variable, matirx form. torch tensor
    q = X_struct.q
    q.requires_grad_(True)
    
    # obtain data_in from runExample.py
    m = data_in.m
    Y = data_in.Y
    
    # objective function
    qtY = q.t() @ Y
    f = 1/m * torch.norm(qtY, p = 1)

    # inequality constraint, matrix form
    ci = None

    # equality constraint 
    ce = GeneralStruct()
    ce.c1 = q.t() @ q - 1

    return [f,ci,ce]