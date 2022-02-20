from tkinter.messagebox import NO
import numpy as np
import torch
from pygranso.private.getObjGrad import getObjGradDL,getObjGrad
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.vec2tensor import vec2tensor
from pygranso.private.getCiVec import getCiVec
from pygranso.private.getCiGradVec import getCiGradVec,getCiGradVecDL
import traceback,sys

def tensor2vec(combinedFunction,x,var_dim_map,nvar,  torch_device = torch.device('cpu'), model = None, double_precision=True, get_grad = True, globalAD = True):
    """
    tensor2vec
        Return vector form objective and constraints information required by PyGRANSO

        USAGE:
            [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec] =
            tensor2vec(combinedFunction,x,var_dim_map,nvar,
                        torch_device = torch.device('cpu'), model = None, double_precision=True)

        INPUT:
            combinedFunction:
                    Function handle of single input X, a data structuture storing all input variables,
                    for evaluating:

                    - The values of the objective and
                        constraints simultaneously:
                        [f,ci,ce] = combinedFunction(X)
                        In this case, ci and/or ce should be returned as
                        None if no (in)equality constraints are given.

            var_dim_map:

                        A dictionary for optmization variable information,
                        where the key is the variable name and val is a list for correpsonding dimension:
                        e.g., var_in = {"x": [1,1]}; var_in = {"U": [5,10], "V": [10,20]}

                        It should not be used when nn_model is specfied, as optimization variable information can be
                        obtained from neural network model


            x:
                    Vector, i.e., n by 1 torch tensor form optimization variables.
                    This vector is detached from the computational graph of ci_vec_torch

            nvar:
                    Total number of optimization variables

            torch_device:

                    Default: torch.device('cpu')

                    Choose torch.device used for matrix operation in PyGRANSO.
                    torch_device = torch.device('cuda') if one wants to use cuda device

            model:
                    Default: None

                    Neural network model defined by torch.nn. It only used when torch.nn was used to
                    define the combinedFunction and/or objEvalFunction

            double_precision

                    Float precision used in PyGRANSO: torch.float or torch.double

        OUTPUT:

            f_vec
                    Scalar form objective function value

            f_grad_vec
                    Vector, i.e., n by 1 torch tensor form gradients of objective.

            ci_vec
                    Vector, i.e., p1 by 1 torch tensor form inequality constraints.

            ci_grad_vec
                    Matrix, i.e., p1 by n torch tensor form gradient of inequality constraints.

            ce_vec
                    Vector, i.e., p2 by 1 torch tensor form equality constraints.

            ce_grad_vec
                    Matrix, i.e., p2 by n torch tensor form gradient of equality constraints.

        If you publish work that uses or refers to PyGRANSO, please cite both
        PyGRANSO and GRANSO paper:

        [1] Buyun Liang, Tim Mitchell, and Ju Sun,
            NCVX: A User-Friendly and Scalable Package for Nonconvex
            Optimization in Machine Learning, arXiv preprint arXiv:2111.13984 (2021).
            Available at https://arxiv.org/abs/2111.13984

        [2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton,
            A BFGS-SQP method for nonsmooth, nonconvex, constrained
            optimization and its evaluation using relative minimization
            profiles, Optimization Methods and Software, 32(1):148-181, 2017.
            Available at https://dx.doi.org/10.1080/10556788.2016.1208749

        tensor2vec.py (introduced in PyGRANSO v1.0.0)
        Copyright (C) 2021 Buyun Liang and Tim Mitchell

        New code and functionality for PyGRANSO v1.0.0.

        For comments/bug reports, please visit the PyGRANSO webpage:
        https://github.com/sun-umn/PyGRANSO

        =========================================================================
        |  PyGRANSO: A PyTorch-enabled port of GRANSO with auto-differentiation |
        |  Copyright (C) 2021 Tim Mitchell and Buyun Liang                      |
        |                                                                       |
        |  This file is part of PyGRANSO.                                       |
        |                                                                       |
        |  PyGRANSO is free software: you can redistribute it and/or modify     |
        |  it under the terms of the GNU Affero General Public License as       |
        |  published by the Free Software Foundation, either version 3 of       |
        |  the License, or (at your option) any later version.                  |
        |                                                                       |
        |  PyGRANSO is distributed in the hope that it will be useful,          |
        |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
        |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
        |  GNU Affero General Public License for more details.                  |
        |                                                                       |
        |  You should have received a copy of the GNU Affero General Public     |
        |  License along with this program.  If not, see                        |
        |  <http://www.gnu.org/licenses/agpl.html>.                             |
        =========================================================================
    """
    if var_dim_map != None:
        X = vec2tensor(x,var_dim_map)
    else:
        # update model parameters using current x
        torch.nn.utils.vector_to_parameters(x, model.parameters())
        X = model

    # obtain objective and constraint function and their corresponding gradient
    # matrix form functions
    try:
        if get_grad:
            if globalAD == False:
                # No auto-differentiation used here
                [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec] = combinedFunction(X)
            else:
                if isinstance(var_dim_map,dict):
                    for var_name in X.__dict__:
                        var = getattr(X,var_name)
                        var.requires_grad_(True)

                [f,ci,ce] = combinedFunction(X)
                [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec] = getValwithAD(X,f,ci,ce,var_dim_map,nvar, torch_device, model, double_precision)

            return [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec]
        else:
            # Not return grad information. Used in back tracking line-search
            if globalAD == False:
                [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec] = combinedFunction(X)
            else:
                [f,ci,ce] = combinedFunction(X)
                [f_vec,ci_vec,ce_vec] = getVal4LineSearch(f,ci,ce,torch_device, double_precision)

            return [f_vec,ci_vec,ce_vec]
    except Exception as e:
        print(traceback.format_exc())
        print("Please check the setting of opts.globalAD")
        sys.exit()

def getValwithAD(X,f,ci,ce,var_dim_map,nvar, torch_device, model, double_precision):
    """
    getValwithAD:
        getValwithAD obtains vector form information [f,f_grad,ci,ci_grad,ce,ce_grad] 
        with autodifferentiation
    """
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
        if model == None:
            ci_grad_vec = getCiGradVec(nvar,nconstr_ci_total,var_dim_map,X,ci_vec_torch,torch_device,double_precision)
        else:
            ci_grad_vec = getCiGradVecDL(nvar,nconstr_ci_total,model,ci_vec_torch, torch_device, double_precision)
        # print(ci_grad_vec)
    else:
        ci_vec = None
        ci_grad_vec = None

    ##  ce and ce_grad
    if ce != None:
        [ce_vec,ce_vec_torch,nconstr_ce_total] = getCiVec(ce,torch_device,double_precision)
        if model == None:
            ce_grad_vec = getCiGradVec(nvar,nconstr_ce_total,var_dim_map,X,ce_vec_torch,torch_device,double_precision)
        else:
            ce_grad_vec = getCiGradVecDL(nvar,nconstr_ce_total,model,ce_vec_torch, torch_device, double_precision)

    else:
        ce_vec = None
        ce_grad_vec = None

    return [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec]


def getVal4LineSearch(f,ci,ce,torch_device, double_precision):
    """
    getVal4LineSearch:
        getValwithAD obtains vector form information [f,ci,ce] 
    """
        # obj function is a scalar form
    try:
        f_vec = f.item()
    except Exception:
        f_vec = f

    ##  ci and ci_grad
    if ci != None:
        [ci_vec,_,_] = getCiVec(ci,torch_device,double_precision)
    else:
        ci_vec = None

    ##  ce and ce_grad
    if ce != None:
        [ce_vec,_,_] = getCiVec(ce,torch_device,double_precision)
    else:
        ce_vec = None

    return [f_vec,ci_vec,ce_vec]