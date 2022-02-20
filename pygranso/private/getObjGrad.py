import numpy as np
import torch


def getObjGrad(nvar,var_dim_map,f,X, torch_device,double_precision):
    """
    getObjGrad:
        getObjGrad obtains gradient of objective function by using pytorch autodiff

        USAGE:
            f_grad_vec = getObjGrad(nvar,var_dim_map,f,X, torch_device,double_precision)

        INPUT:
            nvar
                    Number of optimization variables in total

            var_dim_map

                A dictionary for optmization variable information,
                where the key is the variable name and val is a list for correpsonding dimension:
                e.g., var_in = {"x": [1,1]}; var_in = {"U": [5,10], "V": [10,20]}

            f

                Function handle of objective function

            X

                A data structuture storing all input variables

            torch_device

                    Choose torch.device used for matrix operation in PyGRANSO.
                    torch_device = torch.device('cuda') if one wants to use cuda device

            double_precision

                    float precision used in PyGRANSO, torch.float or torch.double

        OUTPUT:

            f_grad_vec
                    Vector, i.e., n by 1 torch tensor form gradients of objective function.

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

        getObjGrad.py (introduced in PyGRANSO v1.0.0)
        Copyright (C) 2021 Buyun Liang

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
    
    if double_precision:
        torch_dtype = torch.double
    else:
        torch_dtype = torch.float
    
    try:
        f.backward(retain_graph=True)
        

        f_grad_vec = torch.zeros((nvar,1),device=torch_device, dtype=torch_dtype)

        curIdx = 0
        # current variable, e.g., U
        for var in var_dim_map.keys():
            # corresponding dimension of the variable, e.g, 3 by 2
            dim = var_dim_map.get(var)
            grad_tmp = getattr(X,var).grad
            varLen = np.prod(dim)
            if grad_tmp == None:
                curIdx += varLen
                continue
            f_grad_reshape = torch.reshape(grad_tmp,(varLen,1))
            f_grad_vec[curIdx:curIdx+varLen] = f_grad_reshape
            curIdx += varLen

            # preventing gradient accumulating
            getattr(X,var).grad.zero_()
    except Exception:
        f_grad_vec = torch.zeros((nvar,1),device=torch_device, dtype=torch_dtype)
    f_grad_vec = f_grad_vec.detach()
    return f_grad_vec

def getObjGradDL(nvar,model,f, torch_device, double_precision):
    """
    getObjGradDL:
        getObjGrad obtains gradient of objective function defined by torch.nn module
        by using pytorch autodiff

        USAGE:

            f_grad_vec = getObjGradDL(nvar,model,f, torch_device, double_precision)

        INPUT:

            nvar
                    Number of optimization variables in total

            model

                torch.nn model used to define the neural network in the problem

            f

                Function handle of objective function

            torch_device

                    Choose torch.device used for matrix operation in PyGRANSO.
                    torch_device = torch.device('cuda') if one wants to use cuda device

            double_precision

                    float precision used in PyGRANSO, torch.float or torch.double

        OUTPUT:

            f_grad_vec

                    Vector, i.e., n by 1 torch tensor form gradients of objective function.


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

        getObjGrad.py (introduced in PyGRANSO v1.0.0)
        Copyright (C) 2021 Buyun Liang

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
    f.backward(retain_graph=True)
    # transform f_grad form matrix form to vector form
    if double_precision:
        torch_dtype = torch.double
    else:
        torch_dtype = torch.float

    f_grad_vec = torch.zeros((nvar,1),device=torch_device, dtype=torch_dtype)

    curIdx = 0
    parameter_lst = list(model.parameters())
    for i in range(len(parameter_lst)):
        f_grad_reshape = torch.reshape(parameter_lst[i].grad,(-1,1))
        f_grad_vec[curIdx:curIdx+f_grad_reshape.shape[0]] = f_grad_reshape
        curIdx += f_grad_reshape.shape[0]
        # preventing gradient accumulating
        parameter_lst[i].grad.zero_()
        
    f_grad_vec = f_grad_vec.detach()
    return f_grad_vec
