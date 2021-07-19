import numpy as np
from pygransoStruct import genral_struct

def getObjGrad(nvar,var_dim_map,f,X):
    f_grad = genral_struct()
    f.backward()
    # current variable, e.g., U
    for var in var_dim_map.keys():
        grad_tmp = getattr(X,var).grad
        setattr(f_grad,var,grad_tmp)  
      # transform f_grad form matrix form to vector form
    f_grad_vec = np.zeros((nvar,1))
    curIdx = 0
    # current variable, e.g., U
    for var in var_dim_map.keys():
        # corresponding dimension of the variable, e.g, 3 by 2
        tmpDim1 = var_dim_map.get(var)[0]
        tmpDim2 = var_dim_map.get(var)[1]
        
        f_grad_vec[curIdx:curIdx+tmpDim1*tmpDim2] = np.reshape(getattr(f_grad,var),(tmpDim1*tmpDim2,1))
        curIdx += tmpDim1 * tmpDim2
    return f_grad_vec