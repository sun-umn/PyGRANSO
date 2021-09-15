from pygransoStruct import general_struct
import torch
from torch import linalg as LA

def eval_obj(X_struct,data_in = None):
    # user defined variable, matirx form. torch tensor
    X = X_struct.X
    X.requires_grad_(True)
    
    # obtain data_in from runExample.py
    A = data_in.A
    B = data_in.B
    C = data_in.C
    # stability_margin = data_in.stability_margin

    # objective function
    M           = A + B@X@C
    [D,_]       = LA.eig(M)
    f = torch.max(D.imag)
    return f

def combinedFunction(X_struct,data_in = None):
    
    # user defined variable, matirx form. torch tensor
    X = X_struct.X
    X.requires_grad_(True)
    
    # obtain data_in from runExample.py
    A = data_in.A
    B = data_in.B
    C = data_in.C
    stability_margin = data_in.stability_margin

    # objective function
    M           = A + B@X@C
    [D,_]       = LA.eig(M)
    f = torch.max(D.imag)

    # inequality constraint, matrix form
    ci = general_struct()
    ci.c1 = torch.max(D.real) + stability_margin

    # equality constraint 
    ce = None
    
    return [f,ci,ce]