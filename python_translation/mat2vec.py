def mat2vec(x,var_dim_map,nvar,parameters = None):


    # input variables (matrix form), e.g., {'U','V'};
    var = var_dim_map.keys()
    # corresponding dimensions (matrix form), e.g., {[3,2],[4,2]};
    dim = var_dim_map.values()
    # reshape vector input x to matrix form X, e.g., X.U and X.V
    curIdx = 0
    for idx in range(len(var)):
        # current variable, e.g., U
        tmpVar = var[idx]
        # corresponding dimension of the variable, e.g, 3 by 2
        tmpDim1 = dim[idx][0]
        tmpDim2 = dim[idx][1]
        # reshape vector input x in to matrix variables, e.g, X.U, X.V
        curIdx += 1
        X.(tmpVar) = reshape(x(curIdx:curIdx+tmpDim1*tmpDim2-1),tmpDim1,tmpDim2);
        curIdx = curIdx+tmpDim1*tmpDim2-1;
    

    f_vec = 1
    f_grad_vec = 1
    ci_vec = 1
    ci_grad_vec = 1
    ce_vec = 1
    ce_grad_vec = 1

    return [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec]