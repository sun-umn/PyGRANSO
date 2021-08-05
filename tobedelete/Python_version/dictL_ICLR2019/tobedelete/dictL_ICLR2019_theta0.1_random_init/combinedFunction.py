import numpy as np
from numpy import linalg as LA
from numpy.core.fromnumeric import transpose
from pygransoStruct import general_struct
import torch

class ciStruct:
    pass


def combinedFunction(X):
    
    # input variable, matirx form. torch tensor
    q = X.q
    q.requires_grad_(True)
    
    n = 400
    m = 10*n**2

    theta = 0.1   # sparsity level

    torch.manual_seed(2021)
    Y = torch.rand(n,m) * (torch.rand(n,m) <= theta) # Bernoulli-Gaussian model
    
    # objective function
    qtY = torch.mm( torch.transpose(q,0,1).double(), Y.double())
    f = 1/m * torch.norm(qtY, p = 1)

    # inequality constraint, matrix form
    ci = None

    # equality constraint 
    ce = general_struct()
    ce.c1 = torch.norm(q, p = 2) - 1

    return [f,ci,ce]