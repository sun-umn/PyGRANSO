import numpy as np
from numpy import linalg as LA
from numpy.core.fromnumeric import transpose
from pygransoStruct import general_struct
import torch
import scipy.io
# import h5py


class ciStruct:
    pass


def combinedFunction(X,parameters):
    
    # input variable, matirx form. torch tensor
    q = X.q
    q.requires_grad_(True)
    
    m = parameters.m
    Y = parameters.Y
    
    # objective function
    qtY = q.t().double() @ Y.double()
    f = 1/m * torch.norm(qtY, p = 1)

    # inequality constraint, matrix form
    ci = None

    # equality constraint 
    ce = general_struct()
    # ce.c1 = torch.norm(q, p = 2) - 1
    # ce.c1  = torch.mm( torch.transpose(q,0,1).double(), q.double()) - 1
    ce.c1 = q.t().double() @ q.double() - 1

    return [f,ci,ce]