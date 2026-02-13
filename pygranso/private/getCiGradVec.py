import numpy as np
import torch


def getCiGradVec(
    nvar, nconstr_ci_total, var_dim_map, X, ci_vec_torch, torch_device, double_precision
):
    """
    getCiGradVec:
        getCiGradVec obtains gradient of constraints function (in the vector form) by using pytorch autodiff

        USAGE:
            ci_grad_vec = getCiGradVec(nvar,nconstr_ci_total,var_dim_map,X,ci_vec_torch, torch_device, double_precision)

        INPUT:
            nvar
                    number of (scalar form) optimization variables in total

            nconstr_ci_total

                    number of inequality OR equality constraints

            var_dim_map

                    A dictionary for optmization variable information,
                    where the key is the variable name and val is a list for correpsonding dimension:
                    e.g., var_in = {"x": [1,1]}; var_in = {"U": [5,10], "V": [10,20]}

                    It should not be used when nn_model is specfied, as optimization variable information can be
                    obtained from neural network model

            X
                    A data structuture storing all input variables

            ci_vec_torch

                    Vector, i.e., p by 1 torch tensor form inequality OR equality constraints

            torch_device

                    Choose torch.device used for matrix operation in PyGRANSO.
                    torch_device = torch.device('cuda') if one wants to use cuda device

            double_precision

                    float precision used in PyGRANSO, torch.float or torch.double

        OUTPUT:
            ci_grad_vec

                    Vector, i.e., p by 1 torch tensor form gradients of inequality OR equality constraints

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

        getCiGradVec.py (introduced in PyGRANSO v1.0.0)
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
    # gradient of inquality constraints
    if double_precision:
        torch_dtype = torch.double
    else:
        torch_dtype = torch.float

    ci_grad_vec = torch.zeros(
        (nvar, nconstr_ci_total), device=torch_device, dtype=torch_dtype
    )

    # OPTIMIZED: Batch gradient computation using torch.autograd.grad
    # This is much faster than sequential .backward() calls
    var_list = [getattr(X, var) for var in var_dim_map.keys()]

    for i in range(nconstr_ci_total):
        # Compute gradients for all variables at once for this constraint
        grads = torch.autograd.grad(
            ci_vec_torch[i],
            var_list,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        # Flatten and concatenate all gradients
        curIdx = 0
        for var, grad_tensor in zip(var_dim_map.keys(), grads):
            dim = var_dim_map.get(var)
            if grad_tensor is not None:
                varLen = np.prod(dim)
                ci_grad_reshape = grad_tensor.reshape(-1)
                ci_grad_vec[curIdx : curIdx + varLen, i] = ci_grad_reshape
            curIdx += np.prod(dim)

    ci_grad_vec = ci_grad_vec.detach()
    return ci_grad_vec


def getCiGradVecDL(
    nvar, nconstr_ci_total, model, ci_vec_torch, torch_device, double_precision
):
    """
    TODO


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

    ci_grad_vec = torch.zeros(
        (nvar, nconstr_ci_total), device=torch_device, dtype=torch_dtype
    )

    # OPTIMIZED: Batch gradient computation using torch.autograd.grad
    # This is much faster than sequential .backward() calls
    parameter_lst = list(model.parameters())

    for i in range(nconstr_ci_total):
        # Compute gradients for all parameters at once for this constraint
        grads = torch.autograd.grad(
            ci_vec_torch[i],
            parameter_lst,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        # Flatten and concatenate all gradients
        curIdx = 0
        for parameter, grad_tensor in zip(parameter_lst, grads):
            varLen = torch.numel(parameter)
            if grad_tensor is not None:
                ci_grad_reshape = grad_tensor.reshape(-1)
                ci_grad_vec[curIdx : curIdx + varLen, i] = ci_grad_reshape
            curIdx += varLen

    ci_grad_vec = ci_grad_vec.detach()
    return ci_grad_vec
