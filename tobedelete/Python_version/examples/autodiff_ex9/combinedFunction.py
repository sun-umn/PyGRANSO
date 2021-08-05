import numpy as np
from numpy import linalg as LA
from numpy.core.fromnumeric import transpose
import torch

class f_gradStruct:
    pass

class ciStruct:
    pass

class ci_gradStruct:
    class c1:
        pass
    class c2:
        pass

def combinedFunction(X):
    
    # input variable, matirx form
    U = X.U
    V = X.V
    U.requires_grad_(True)
    V.requires_grad_(True)

    D = torch.tensor([[1, 1, 1, 1],[1, 1, 1, 1],[1, 1, 1, 1]])
    D_UV = D - torch.mm(U,torch.transpose(V, 0, 1)) # transpose(Tensor input, int dim0, int dim1)
    # objective function
    f = .5 * torch.norm(D_UV, p = 1)**2

    # inequality constraint, matrix form
    ci = ciStruct()
    ci.c1 = -U 
    ci.c2 = -V  

    # equality constraint 
    ce = None

    return [f,ci,ce]