from pygransoStruct import general_struct
import torch

def combinedFunction(X_struct,parameters):
    
    # user defined variable, matirx form. torch tensor
    q = X_struct.q
    q.requires_grad_(True)
    
    # obtain parameters from runExample.py
    m = parameters.m
    Y = parameters.Y
    
    # objective function
    qtY = q.t() @ Y
    f = 1/m * torch.norm(qtY, p = 1)

    # inequality constraint, matrix form
    ci = None

    # equality constraint 
    ce = general_struct()
    ce.c1 = q.t() @ q - 1

    return [f,ci,ce]