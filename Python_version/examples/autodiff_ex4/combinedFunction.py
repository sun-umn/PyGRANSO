import numpy as np
from torch import linalg as LA
from numpy.core.fromnumeric import transpose
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
import scipy.io
import scipy.linalg as la
from numpy import conjugate as conj
import torch

# print(torch.__version__)

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
    XX           = X.XX
    XX.requires_grad_(True)

    # read input data from matlab file
    file = currentdir + "/ex4_data_n=200.mat"
    mat = scipy.io.loadmat(file)
    mat_struct = mat['sys']
    val = mat_struct[0,0]
    A = torch.from_numpy(val['A'])
    B = torch.from_numpy(val['B'])
    C = torch.from_numpy(val['C'])

    stability_margin = 1
    M           = A + B@XX@C
    [D,V]       = LA.eig(M)
    d           = D
    f = torch.max(d.imag)
    
    #  INEQUALITY CONSTRAINT AT X
    #  Compute the spectral abscissa of A from the spectrum and an
    #  eigenvalue associated with it
    ci = ciStruct()
    ci.c1          = torch.max(d.real) + stability_margin

    #  EQUALITY CONSTRAINT
    #  Return [] when (in)equality constraints are not present.
    ce = None
    # ce_grad = None

    return [f,ci,ce]