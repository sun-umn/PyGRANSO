from dbg_print import dbg_print_1
import numpy as np
import torch

def getCiGradVec(nvar,nconstr_ci_total,var_dim_map,X,ci_vec_torch, torch_device):

    dbg_print_1('Using device in getObjGrad')
    # gradient of inquality constraints
    ci_grad_vec = torch.zeros((nvar,nconstr_ci_total),device=torch_device, dtype=torch.double)
    
    for i in range(nconstr_ci_total):
        ci_vec_torch[i].backward(retain_graph=True)
        # current variable, e.g., U
        curIdx = 0
        for var in var_dim_map.keys():
            # corresponding dimension of the variable, e.g, 3 by 2
            dim = var_dim_map.get(var) 
            ci_grad_tmp = getattr(X,var).grad
            varLen = np.prod(dim)
            ci_grad_reshape = torch.reshape(ci_grad_tmp,(varLen,1))[:,0]
            ci_grad_vec[curIdx:curIdx+varLen,i] = ci_grad_reshape 
            curIdx += varLen
            getattr(X,var).grad.zero_()

    return ci_grad_vec