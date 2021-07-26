import numpy as np
from numpy import linalg as LA
from numpy.core.fromnumeric import transpose
from pygransoStruct import general_struct
import torch

class ciStruct:
    pass


def combinedFunction(X):
    
    # input variable, matirx form. torch tensor
    x = X.x
    x.requires_grad_(True)
    
    n = 1000

    lambda_1 = 0.5
    torch.manual_seed(2021)
    b = torch.rand(n,1)
    pos_one = torch.ones(n-1)
    neg_one = -torch.ones(n-1)
    F = torch.zeros(n-1,n)
    F[:,0:n-1] += torch.diag(neg_one,0) 
    F[:,1:n] += torch.diag(pos_one,0)
    # print(F)

    # objective function
    f = torch.norm(x-b, p = 2)**2 
    Fx = torch.mm(F.to(torch.float64),x)
    f += lambda_1 * torch.norm( Fx, p = 1)

    # inequality constraint, matrix form
    ci = None

    # equality constraint 
    ce = None

    return [f,ci,ce]