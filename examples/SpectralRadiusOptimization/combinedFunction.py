from pygransoStruct import general_struct
import torch
from torch import linalg as LA


def combinedFunctionDL(X_struct,parameters = None):
    
    # user defined variable, matirx form. torch tensor
    X = X_struct.X
    X.requires_grad_(True)
    
    # obtain parameters from runExample.py
    A = parameters.A
    B = parameters.B
    C = parameters.C
    stability_margin = parameters.stability_margin

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