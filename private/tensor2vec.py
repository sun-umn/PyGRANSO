import numpy as np
import torch
from private.getObjGrad import getObjGradDL,getObjGrad
from private.vec2tensor import vec2tensor
from private.getCiVec import getCiVec
from private.getCiGradVec import getCiGradVec

def obj_eval(eval_obj, x, var_dim_map, data_in = None):
    """
    obj_eval makes an objective evaluation function used for backtrack line search
    """
    X_struct = vec2tensor(x,var_dim_map)
    if data_in == None:
        f = eval_obj(X_struct)
    else:
        f = eval_obj(X_struct,data_in)
    return f

def tensor2vec(combinedFunction,x,var_dim_map,nvar,data_in = None,  torch_device = torch.device('cpu'), model = None, double_precision=True):
    """
    mat2vec_autodiff
        Return vector form objective and constraints information required by PyGRANSO
    """
    X = vec2tensor(x,var_dim_map)
    # obtain objective and constraint function and their corresponding gradient
    # matrix form functions    
    
    if data_in == None:
        [f,ci,ce] = combinedFunction(X)
    else:
        [f,ci,ce] = combinedFunction(X,data_in)
        
    # obj function is a scalar form
    try: 
        f_vec = f.item()    
    except Exception:
        f_vec = f
    
    if model == None:
    # if True:
        f_grad_vec = getObjGrad(nvar,var_dim_map,f,X,torch_device,double_precision)
    else:
        f_grad_vec = getObjGradDL(nvar,model,f, torch_device, double_precision)

    ##  ci and ci_grad
    if ci != None:
        [ci_vec,ci_vec_torch,nconstr_ci_total] = getCiVec(ci,torch_device,double_precision)
        ci_grad_vec = getCiGradVec(nvar,nconstr_ci_total,var_dim_map,X,ci_vec_torch,torch_device,double_precision)
        # print(ci_grad_vec)
    else:
        ci_vec = None
        ci_grad_vec = None

    ##  ce and ce_grad
    if ce != None:
        [ce_vec,ce_vec_torch,nconstr_ce_total] = getCiVec(ce,torch_device,double_precision)
        ce_grad_vec = getCiGradVec(nvar,nconstr_ce_total,var_dim_map,X,ce_vec_torch,torch_device,double_precision)
        
    else:
        ce_vec = None
        ce_grad_vec = None

    return [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec]


