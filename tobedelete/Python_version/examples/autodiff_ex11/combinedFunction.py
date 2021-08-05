import numpy as np
from numpy import linalg as LA
from numpy.core.fromnumeric import transpose
from pygransoStruct import general_struct
import torch

class ciStruct:
    pass


def combinedFunction(X):
    
    # input variable, matirx form. torch tensor
    M = X.M
    S = X.S
    M.requires_grad_(True)
    S.requires_grad_(True)
    
    d1 = 5
    d2 = 10
    lambda_1 = 0.5
    torch.manual_seed(2021)
    Y = torch.rand(d1,d2) 
    
    # objective function
    f = torch.norm(M, p = 'nuc') + lambda_1 * torch.norm(S, p = 1)

    # inequality constraint, matrix form
    ci = None

    # equality constraint 
    ce = general_struct()
    ce.c1 = M + S - Y

    return [f,ci,ce]