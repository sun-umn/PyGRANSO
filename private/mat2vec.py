import numpy as np
from combinedFunction import combinedFunctionDL
from pygransoStruct import VariableStruct, general_struct
import torch
from private.getObjGrad import getObjGradDL,getObjGrad
from private.vec2mat import vec2mat
from private.getCiVec import getCiVec
from private.getCiGradVec import getCiGradVec

def mat2vec_autodiff(x,var_dim_map,nvar,parameters = None):
    X = vec2mat(x,var_dim_map)
    # obtain objective and constraint function and their corresponding gradient
    # matrix form functions    
    
    if parameters == None:
        [f,ci,ce] = combinedFunctionDL(X)
    else:
        [f,ci,ce] = combinedFunctionDL(X,parameters)
        
    # obj function is a scalar form
    f_vec = f.item()    
    f_grad_vec = getObjGrad(nvar,var_dim_map,f,X)

    ##  ci and ci_grad
    if ci != None:
        [ci_vec,ci_vec_torch,nconstr_ci_total] = getCiVec(ci)
        ci_grad_vec = getCiGradVec(nvar,nconstr_ci_total,var_dim_map,X,ci_vec_torch)
        # print(ci_grad_vec)
    else:
        ci_vec = None
        ci_grad_vec = None

    ##  ce and ce_grad
    if ce != None:
        [ce_vec,ce_vec_torch,nconstr_ce_total] = getCiVec(ce)
        ce_grad_vec = getCiGradVec(nvar,nconstr_ce_total,var_dim_map,X,ce_vec_torch)
        
    else:
        ce_vec = None
        ce_grad_vec = None

    return [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec]


def tensor2vec_autodiff(x,model,nvar,parameters = None):

    torch.nn.utils.vector_to_parameters(x, model.parameters()) # update model paramters
    
    # obtain objective and constraint function and their corresponding gradient
    # matrix form functions    
    
    if parameters == None:
        [f,ci,ce] = combinedFunctionDL(model)
    else:
        [f,ci,ce] = combinedFunctionDL(model,parameters)
        
    # obj function is a scalar form
    f_vec = f.item()    
    f_grad_vec = getObjGradDL(nvar,model,f)

    # print("\n\n\n\n\nPrint Gradient\n\n\n\n\n")

    # lst = list(model.parameters())

    # for i in range(len(lst)):
    #         print(lst[i].grad.shape)
    #         print(lst[i].grad[0])

    

    ##  ci and ci_grad
    if ci != None:
        [ci_vec,ci_vec_torch,nconstr_ci_total] = getCiVec(ci)
        ci_grad_vec = getCiGradVec(nvar,nconstr_ci_total,var_dim_map,X,ci_vec_torch)
        # print(ci_grad_vec)
    else:
        ci_vec = None
        ci_grad_vec = None

    ##  ce and ce_grad
    if ce != None:
        [ce_vec,ce_vec_torch,nconstr_ce_total] = getCiVec(ce)
        ce_grad_vec = getCiGradVec(nvar,nconstr_ce_total,var_dim_map,X,ce_vec_torch)
        
    else:
        ce_vec = None
        ce_grad_vec = None

    return [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec]


