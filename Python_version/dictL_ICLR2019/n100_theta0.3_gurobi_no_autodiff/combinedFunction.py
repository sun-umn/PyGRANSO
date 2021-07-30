import numpy as np
# from numpy import linalg as LA
from numpy.core.fromnumeric import transpose
from pygransoStruct import general_struct
import torch
import numpy.linalg as LA
# import h5py


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
    
    # input variable, matirx form. torch tensor
    q = X.q
    # q.requires_grad_(True)
    
    n = 100
    m = 10*n**2

    theta = 0.3   # sparsity level

    

    np.random.seed(2021)
    Y = np.random.randn(n,m) * (np.random.rand(n,m) <= theta) # Bernoulli-Gaussian model
    
    
    # objective function
    qtY = q.T @ Y
    f = 1/m * LA.norm(qtY,  1)



    f_grad = general_struct()
    f_grad.q = 1/m * Y @ np.sign(qtY.T)


    # inequality constraint, matrix form
    ci = None
    ci_grad = None

    # equality constraint 
    ce = general_struct()
    # ce.c1 = LA.norm(q, ord = 2) - 1
    ce.c1 = q.T @ q - 1

    ce_grad = ci_gradStruct()
    ce_grad.c1.q = 2*q

    return [f,f_grad,ci,ci_grad,ce,ce_grad]