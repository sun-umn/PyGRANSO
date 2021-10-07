from pygransoStruct import general_struct
import numpy as np
import torch

def vec2mat(x,var_dim_map):
    X = general_struct()
    # reshape vector input x to matrix form X, e.g., X.U and X.V
    curIdx = 0
    # current variable, e.g., U
    for var in var_dim_map.keys():
        # corresponding dimension of the variable, e.g, 3 by 2
        dim = var_dim_map.get(var)
        # reshape vector input x in to matrix variables, e.g, X.U, X.V
        tmpMat = torch.reshape(x[curIdx:curIdx + np.prod(dim)],tuple(dim))
        setattr(X, var, tmpMat)
        curIdx += np.prod(dim)
    return X


    
    