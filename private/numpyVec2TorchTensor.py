from pygransoStruct import general_struct
import numpy as np
import torch

def numpyVec2TorchTensor(x,var_dim_map):
    X = general_struct()
    # reshape vector input x to matrix form X, e.g., X.U and X.V
    curIdx = 0
    # current variable, e.g., U
    for var in var_dim_map.keys():
        # corresponding dimension of the variable, e.g, 3 by 2
        tmpDim1 = var_dim_map.get(var)[0]
        tmpDim2 = var_dim_map.get(var)[1]
        # reshape vector input x in to matrix variables, e.g, X.U, X.V
        tmpMat = np.reshape(x[curIdx:curIdx+tmpDim1*tmpDim2],(tmpDim1,tmpDim2))
        setattr(X, var, torch.from_numpy(tmpMat))
        curIdx += tmpDim1 * tmpDim2
    return X

def numpyVec2DLTorchTensor(x,model):
    x_torch = torch.from_numpy(x).cuda()
    torch.nn.utils.vector_to_parameters(x_torch, model.parameters())

    
    