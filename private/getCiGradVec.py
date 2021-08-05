import numpy as np

def getCiGradVec(nvar,nconstr_ci_total,var_dim_map,X,ci_vec_torch):
    # gradient of inquality constraints
    ci_grad_vec = np.zeros((nvar,nconstr_ci_total))
    
    for i in range(nconstr_ci_total):
        ci_vec_torch[i].backward(retain_graph=True)
        # current variable, e.g., U
        curIdx = 0
        for var in var_dim_map.keys():
            # corresponding dimension of the variable, e.g, 3 by 2
            tmpDim1 = var_dim_map.get(var)[0]
            tmpDim2 = var_dim_map.get(var)[1]
            ci_grad_tmp = getattr(X,var).grad.numpy()
            ci_grad_reshape = np.reshape(ci_grad_tmp,(tmpDim1*tmpDim2,1))[:,0]
            ci_grad_vec[curIdx:curIdx+tmpDim1*tmpDim2,i] = ci_grad_reshape 
            curIdx += tmpDim1 * tmpDim2
            getattr(X,var).grad.zero_()

    return ci_grad_vec