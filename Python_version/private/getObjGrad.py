import numpy as np
from pygransoStruct import general_struct

def getObjGrad(nvar,var_dim_map,f,X):
    # f_grad = genral_struct()
    f.backward(retain_graph=True)
    # transform f_grad form matrix form to vector form
    f_grad_vec = np.zeros((nvar,1))

    curIdx = 0
    # current variable, e.g., U
    for var in var_dim_map.keys():
        # corresponding dimension of the variable, e.g, 3 by 2
        tmpDim1 = var_dim_map.get(var)[0]
        tmpDim2 = var_dim_map.get(var)[1]
        grad_tmp = getattr(X,var).grad.numpy()
        f_grad_reshape = np.reshape(grad_tmp,(tmpDim1*tmpDim2,1))
        f_grad_vec[curIdx:curIdx+tmpDim1*tmpDim2] = f_grad_reshape
        curIdx += tmpDim1 * tmpDim2

        # preventing gradient accumulating
        getattr(X,var).grad.zero_()
    return f_grad_vec