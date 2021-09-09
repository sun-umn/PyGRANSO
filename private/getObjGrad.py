from dbg_print import dbg_print_1
import numpy as np
from pygransoStruct import general_struct
import torch

def getObjGrad(nvar,var_dim_map,f,X):
    # f_grad = genral_struct()
    f.backward(retain_graph=True)
    # transform f_grad form matrix form to vector form
    # f_grad_vec = np.zeros((nvar,1))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dbg_print_1('Using device in getObjGrad')
    f_grad_vec = torch.zeros(nvar,1).to(device=device, dtype=torch.double)

    curIdx = 0
    # current variable, e.g., U
    for var in var_dim_map.keys():
        # corresponding dimension of the variable, e.g, 3 by 2
        tmpDim1 = var_dim_map.get(var)[0]
        tmpDim2 = var_dim_map.get(var)[1]
        grad_tmp = getattr(X,var).grad
        f_grad_reshape = torch.reshape(grad_tmp,(tmpDim1*tmpDim2,1))
        f_grad_vec[curIdx:curIdx+tmpDim1*tmpDim2] = f_grad_reshape
        curIdx += tmpDim1 * tmpDim2

        # preventing gradient accumulating
        getattr(X,var).grad.zero_()
    return f_grad_vec

def getObjGradDL(nvar,model,f):
    # f_grad = genral_struct()
    f.backward()
    # transform f_grad form matrix form to vector form
    f_grad_vec = np.zeros((nvar,1))

    curIdx = 0
    parameter_lst = list(model.parameters())
    for i in range(len(parameter_lst)):
        # print(parameter_lst[i].grad.shape)
        f_grad_reshape = torch.reshape(parameter_lst[i].grad,(-1,1)).cpu().numpy()
        f_grad_vec[curIdx:curIdx+f_grad_reshape.shape[0]] = f_grad_reshape
        curIdx += f_grad_reshape.shape[0]

    return f_grad_vec