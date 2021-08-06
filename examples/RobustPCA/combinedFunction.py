from pygransoStruct import general_struct
import torch

def combinedFunction(X_struct,parameters = None):
    
    # user defined variable, matirx form. torch tensor
    M = X_struct.M
    S = X_struct.S
    M.requires_grad_(True)
    S.requires_grad_(True)
    
    # obtain parameters from runExample.py
    eta = parameters.eta
    Y = parameters.Y
    
    # objective function
    f = torch.norm(M, p = 'nuc') + eta * torch.norm(S, p = 1)

    # inequality constraint, matrix form
    ci = None

    # equality constraint 
    ce = general_struct()
    ce.c1 = M + S - Y

    return [f,ci,ce]