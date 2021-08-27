import numpy as np
from combinedFunction import combinedFunctionDL
from pygransoStruct import VariableStruct, general_struct
import torch
from private.getObjGrad import getObjGradDL
from private.numpyVec2TorchTensor import numpyVec2TorchTensor, numpyVec2DLTorchTensor
from private.getCiVec import getCiVec
from private.getCiGradVec import getCiGradVec

def mat2vec(x,var_dim_map,nvar,parameters = None):

    ################################################################################

    X = VariableStruct()
    # reshape vector input x to matrix form X, e.g., X.U and X.V
    curIdx = 0
    # current variable, e.g., U
    for var in var_dim_map.keys():
        # corresponding dimension of the variable, e.g, 3 by 2
        tmpDim1 = var_dim_map.get(var)[0]
        tmpDim2 = var_dim_map.get(var)[1]
        # reshape vector input x in to matrix variables, e.g, X.U, X.V
        tmpMat = np.reshape(x[curIdx:curIdx+tmpDim1*tmpDim2],(tmpDim1,tmpDim2))
        setattr(X, var, tmpMat)
        curIdx += tmpDim1 * tmpDim2

    ################################################################################

    # obtain objective and constraint function and their corresponding gradient

    # matrix form functions
    if parameters == None:
        [f,f_grad,ci,ci_grad,ce,ce_grad] = combinedFunction(X)
    else:
        [f,f_grad,ci,ci_grad,ce,ce_grad] = combinedFunction(X,parameters)
    
    ################################################################################
    # obj function is a scalar form
    f_vec = f

    ################################################################################
    # transform f_grad form matrix form to vector form
    f_grad_vec = np.zeros((nvar,1))
    curIdx = 0
    # current variable, e.g., U
    for var in var_dim_map.keys():
        # corresponding dimension of the variable, e.g, 3 by 2
        tmpDim1 = var_dim_map.get(var)[0]
        tmpDim2 = var_dim_map.get(var)[1]
        
        f_grad_vec[curIdx:curIdx+tmpDim1*tmpDim2] = np.reshape(getattr(f_grad,var),(tmpDim1*tmpDim2,1))
        curIdx += tmpDim1 * tmpDim2

    ################################################################################
    ############################  ci and ci_grad   #################################
    #  number of constraints
    nconstr = 0
    if ci != None:
        # get # of constraints
        # current constraint, e.g., c1, c2
        for constr_i in ci.__dict__.keys():
            constrMatrix = getattr(ci,constr_i)
            nconstr = nconstr + constrMatrix.size
        
        
        # inquality constraints
        ci_vec = np.zeros((nconstr,1))
        curIdx = 0
        # current constraint, e.g., c1, c2
        for constr_i in ci.__dict__.keys():
            constrMatrix = getattr(ci,constr_i)
            ci_vec[curIdx:curIdx + constrMatrix.size] = np.reshape(constrMatrix,(constrMatrix.size,1))
            curIdx = curIdx + constrMatrix.size

        ################################################################################
        # gradient of inquality constraints
        ci_grad_vec = np.zeros((nvar,nconstr))
        # iterate column: constraints
        colIdx = 0
        # current constraint, e.g., c1, c2
        for constr_i in ci.__dict__.keys():
            constrMatrix = getattr(ci,constr_i)
            rowIdx = 0
            # iterate row: variables
            # current variable, e.g., U
            for var in var_dim_map.keys():
                # corresponding dimension of the variable, e.g, 3 by 2
                tmpDim1 = var_dim_map.get(var)[0]
                tmpDim2 = var_dim_map.get(var)[1]
                ciGradMatrix = getattr(getattr(ci_grad,constr_i),var) 
                ci_grad_vec[rowIdx:rowIdx+tmpDim1*tmpDim2, colIdx:colIdx+constrMatrix.size] = ciGradMatrix
                rowIdx += tmpDim1 * tmpDim2
            colIdx += constrMatrix.size
        
    else:
        ci_vec = None
        ci_grad_vec = None


    ################################################################################
    ############################  ce and ce_grad   #################################
    #  number of constraints
    nconstr = 0
    if ce != None:
        # get # of constraints
        # current constraint, e.g., c1, c2
        for constr_i in ce.__dict__.keys():
            constrMatrix = getattr(ce,constr_i)
            nconstr = nconstr + constrMatrix.size
        
        
        # equality constraints
        ce_vec = np.zeros((nconstr,1))
        curIdx = 0
        # current constraint, e.g., c1, c2
        for constr_i in ce.__dict__.keys():
            constrMatrix = getattr(ce,constr_i)
            ce_vec[curIdx:curIdx + constrMatrix.size] = np.reshape(constrMatrix,(constrMatrix.size,1))
            curIdx = curIdx + constrMatrix.size

        ################################################################################
        # gradient of equality constraints
        ce_grad_vec = np.zeros((nvar,nconstr))
        # iterate column: constraints
        colIdx = 0
        # current constraint, e.g., c1, c2
        for constr_i in ce.__dict__.keys():
            constrMatrix = getattr(ce,constr_i)
            rowIdx = 0
            # iterate row: variables
            # current variable, e.g., U
            for var in var_dim_map.keys():
                # corresponding dimension of the variable, e.g, 3 by 2
                tmpDim1 = var_dim_map.get(var)[0]
                tmpDim2 = var_dim_map.get(var)[1]
                ceGradMatrix = getattr(getattr(ce_grad,constr_i),var) 
                ce_grad_vec[rowIdx:rowIdx+tmpDim1*tmpDim2, colIdx:colIdx+constrMatrix.size] = ceGradMatrix
                rowIdx += tmpDim1 * tmpDim2
            colIdx += constrMatrix.size
        
    else:
        ce_vec = None
        ce_grad_vec = None
    
    



    return [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec]



def mat2vec_autodiff(x,var_dim_map,nvar,parameters = None):
    X = numpyVec2TorchTensor(x,var_dim_map)
    # obtain objective and constraint function and their corresponding gradient
    # matrix form functions    
    
    if parameters == None:
        [f,ci,ce] = combinedFunction(X)
    else:
        [f,ci,ce] = combinedFunction(X,parameters)
        
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

    numpyVec2DLTorchTensor(x,model) # update model paramters
    
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