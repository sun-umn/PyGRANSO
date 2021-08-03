import numpy as np
from numpy import linalg as LA
from numpy.core.fromnumeric import transpose
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
import scipy.io
import scipy.linalg as la
from numpy import conjugate as conj

class f_gradStruct:
    pass

class ciStruct:
    pass

class ci_gradStruct:
    class c1:
        pass
    class c2:
        pass

def combinedFunction(X,parameters):
    
    # input variable, matirx form
    XX           = X.XX

    # read input data from matlab file
    # file = currentdir + "/ex4_data_n=200.mat"
    # mat = scipy.io.loadmat(file)
    # mat_struct = mat['sys']
    # val = mat_struct[0,0]
    # A = val['A']
    # B = val['B']
    # p = B.shape[1]
    # C = val['C']
    # m = C.shape[0]

    A = parameters.A
    B = parameters.B
    C = parameters.C
    m = parameters.m
    p = parameters.p

    stability_margin = 1
    M           = A + B@XX@C
    [D,V]       = la.eig(M)
    d           = D
    [D_conj,VL] = la.eig(conj(M.T))
    dl          = conj(D_conj)

    #  OBJECTIVE VALUE AT X
    #  Get the max imaginary part, and an eigenvalue associated with it,
    #  since the constraint is to limit eigenvalues to a band centered on
    #  the x-axis
    d_lst = list(np.imag(d))
    mi   = np.max(d_lst)
    indx        = d_lst.index(mi)
    lambda_var      = d[indx]
    f           = mi

############################################################################
    #  GRADIENT OF THE OBJECTIVE AT X
    #  Get its corresponding right eigenvector
    x           = (V[:,indx]).reshape(p*m,1)
    #  Now find the matching left eigenvector for lambda
    dl_lst = list(np.abs(dl - lambda_var))
    indx    = dl_lst.index(np.min(dl_lst))
    y           = (VL[:,indx]).reshape(p*m,1)
    Bty         = (B.T @ y).reshape(p,1)
    Cx          = (C @ x).reshape(m,1)
    #  Gradient of inner product with respect to A
    #  f_grad.XX is a real-valued matrix, p by m
    f_grad = f_gradStruct()
    f_grad.XX      = np.imag((conj(Bty)@Cx.T)/(conj(y.T)@x))
    # print(sum(sum(f_grad.XX)))
    

    
    #  INEQUALITY CONSTRAINT AT X
    #  Compute the spectral abscissa of A from the spectrum and an
    #  eigenvalue associated with it
    ci = ciStruct()
    d_lst = list(np.real(d))
    ci.c1   = max(d_lst)
    indx        = d_lst.index(ci.c1)
    lambda_var      = d[indx]
    #  account for the stability margin in the inequality constraint
    ci.c1          = ci.c1 + stability_margin



    #  GRADIENT OF THE INEQUALITY CONSTRAINT AT X
    #  Get its corresponding right eigenvector
    ci_grad = ci_gradStruct()
    x           = (V[:,indx]).reshape(p*m,1)
    #  Now find the matching left eigenvector for lambda
    dl_lst = list(abs(dl - lambda_var))
    indx    = dl_lst.index(min(dl_lst))
    y           = (VL[:,indx]).reshape(p*m,1)
    Bty         = (B.T @ y).reshape(p,1)
    Cx          = (C @ x).reshape(m,1)
    #  Gradient of inner product with respect to A
    ci_grad.c1.XX     = np.real((conj(Bty)@Cx.T)/(conj(y.T) @ x))
    # print(sum(sum(ci_grad.c1.XX)))
    ci_grad.c1.XX = (ci_grad.c1.XX).reshape(p*m,1)

    
    #  EQUALITY CONSTRAINT
    #  Return [] when (in)equality constraints are not present.
    ce = None
    ce_grad = None

    return [f,f_grad,ci,ci_grad,ce,ce_grad]