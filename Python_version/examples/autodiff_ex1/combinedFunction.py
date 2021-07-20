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
    x1 = X.x1
    x2 = X.x2
    x1.requires_grad_(True)
    x2.requires_grad_(True)

    # objective function
    # f must be a scalar. not 1 by 1 matrix
    f = (8 * abs(x1**2 - x2) + (1 - x1)**2)[0,0]

    # f_grad = f_gradStruct()
    # #  Compute the 2nd term
    # f_grad.x1      = -2*(1-x1)
    # f_grad.x2 = 0

    # #  Add in the 1st term, where we must handle the sign due to the 
    # #  absolute value
    # if x1**2 - x2 >= 0:
    #   f_grad.x1    = f_grad.x1 + 8*2*x1
    #   f_grad.x2    = f_grad.x2 - 8
    # else:
    #   f_grad.x1    = f_grad.x1 - 8*2*x1
    #   f_grad.x2    = f_grad.x2 + 8
    

    # inequality constraint, matrix form
    ci = ciStruct()
    ci.c1 = np.sqrt(2)*x1-1  
    ci.c2 = 2*x2-1 

    # # gradient of inequality constraint, matrix form
    # ci_grad = ci_gradStruct()
    # # # of constr b # of vars of U and V
    # ci_grad.c1.x1 = np.sqrt(2)
    # ci_grad.c1.x2 = 0
    
    # ci_grad.c2.x1 = 0
    # ci_grad.c2.x2 = 2
    
    # equality constraint 
    ce = None
    # ce_grad = None

    return [f,ci,ce]