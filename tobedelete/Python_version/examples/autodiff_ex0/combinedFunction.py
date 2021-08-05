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
    f = .5 * torch.norm(D_UV, p = 'fro')**2

    # f_grad = f_gradStruct()
    # # gradient of objective function, matrix form
    # f_grad.U = -np.dot(D_UV,V)
    # f_grad.V = -np.dot(np.transpose(D_UV),U)

    # inequality constraint, matrix form
    ci = ciStruct()
    ci.c1 = -U 
    ci.c2 = -V  

    # # gradient of inequality constraint, matrix form
    # ci_grad = ci_gradStruct()
    # ci_grad.c1.U = -np.identity(6)
    # ci_grad.c1.V = np.zeros((8,6))

    # ci_grad.c2.U = np.zeros((6,8))
    # ci_grad.c2.V = -np.identity(8)

    # equality constraint 
    ce = None

    return [f,ci,ce]