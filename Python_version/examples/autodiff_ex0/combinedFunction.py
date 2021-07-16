import numpy as np
from numpy import linalg as LA
from numpy.core.fromnumeric import transpose

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

    D = np.array([[1, 1, 1, 1],[1, 1, 1, 1],[1, 1, 1, 1]])
    D_UV = D - np.dot(U,np.transpose(V))
    # objective function
    f = .5 * LA.norm(D_UV, 'fro')**2

    f_grad = f_gradStruct()
    # gradient of objective function, matrix form
    f_grad.U = -np.dot(D_UV,V)
    f_grad.V = -np.dot(np.transpose(D_UV),U)

    # inequality constraint, matrix form
    ci = ciStruct()
    ci.c1 = -U
    ci.c2 = -V

    # gradient of inequality constraint, matrix form
    ci_grad = ci_gradStruct()
    ci_grad.c1.U = -np.identity(6)
    ci_grad.c1.V = np.zeros((8,6))

    ci_grad.c2.U = np.zeros((6,8))
    ci_grad.c2.V = -np.identity(8)

    # equality constraint 
    ce = None
    ce_grad = None

    return [f,f_grad,ci,ci_grad,ce,ce_grad]