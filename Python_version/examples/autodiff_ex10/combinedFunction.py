import numpy as np
from numpy import linalg as LA
from numpy.core.fromnumeric import transpose
from pygransoStruct import genral_struct
import torch

class ciStruct:
    pass


def combinedFunction(X):
    
    # input variable, matirx form. torch tensor
    q = X.q
    q.requires_grad_(True)
    
    n = 10
    m = 10*n**2
    torch.manual_seed(2021)
    Y = torch.rand(n,m)
    
    # objective function
    qtY = torch.mm( torch.transpose(q,0,1).double(), Y.double())
    f = 1/m * torch.norm(qtY, p = 1)

    # inequality constraint, matrix form
    ci = None

    # equality constraint 
    ce = genral_struct()
    ce.c1 = torch.norm(q, p = 2) - 1

    return [f,ci,ce]