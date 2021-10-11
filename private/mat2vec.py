import numpy as np
from combinedFunction import combinedFunction, eval_obj
from pygransoStruct import VariableStruct, general_struct
import torch
from private.getObjGrad import getObjGradDL,getObjGrad
from private.vec2mat import vec2mat
from private.getCiVec import getCiVec
from private.getCiGradVec import getCiGradVec

def obj_eval_DL(x,model,data_in = None):
    torch.nn.utils.vector_to_parameters(x, model.parameters()) # update model paramters
    
    # obtain objective and constraint function and their corresponding gradient
    # matrix form functions    
    
    if data_in == None:
        f = eval_obj(model)
    else:
        f = eval_obj(model,data_in)
    
    return f

def obj_eval(x, var_dim_map, data_in = None):
    
    X_struct = vec2mat(x,var_dim_map)

    if data_in == None:
        f = eval_obj(X_struct)
    else:
        f = eval_obj(X_struct,data_in)
    
    return f

def mat2vec_autodiff(x,var_dim_map,nvar,data_in = None,  torch_device = torch.device('cpu')):
    X = vec2mat(x,var_dim_map)
    # obtain objective and constraint function and their corresponding gradient
    # matrix form functions    
    
    if data_in == None:
        [f,ci,ce] = combinedFunction(X)
    else:
        [f,ci,ce] = combinedFunction(X,data_in)
        
    # obj function is a scalar form
    f_vec = f.item()    
    if data_in.model == None:
        f_grad_vec = getObjGrad(nvar,var_dim_map,f,X,torch_device)
    else:
        f_grad_vec = getObjGradDL(nvar,data_in.model,f, torch_device)

    ##  ci and ci_grad
    if ci != None:
        [ci_vec,ci_vec_torch,nconstr_ci_total] = getCiVec(ci,torch_device)
        ci_grad_vec = getCiGradVec(nvar,nconstr_ci_total,var_dim_map,X,ci_vec_torch,torch_device)
        # print(ci_grad_vec)
    else:
        ci_vec = None
        ci_grad_vec = None

    ##  ce and ce_grad
    if ce != None:
        [ce_vec,ce_vec_torch,nconstr_ce_total] = getCiVec(ce,torch_device)
        ce_grad_vec = getCiGradVec(nvar,nconstr_ce_total,var_dim_map,X,ce_vec_torch,torch_device)
        
    else:
        ce_vec = None
        ce_grad_vec = None

    return [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec]


def tensor2vec_autodiff(x,model,nvar,data_in = None, torch_device = torch.device('cpu') ):

    torch.nn.utils.vector_to_parameters(x, model.parameters()) # update model paramters
    
    # obtain objective and constraint function and their corresponding gradient
    # matrix form functions    
    
    if data_in == None:
        [f,ci,ce] = combinedFunction(model)
    else:
        [f,ci,ce] = combinedFunction(model,data_in)
        
    # obj function is a scalar form
    f_vec = f.item()    
    f_grad_vec = getObjGradDL(nvar,model,f, torch_device)

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


