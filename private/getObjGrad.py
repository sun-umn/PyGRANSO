import numpy as np
import torch


def getObjGrad(nvar,var_dim_map,f,X, torch_device,double_precision):
    """
    getObjGrad:
        getObjGrad obtains gradient of objective function by using pytorch autodiff

        If you publish work that uses or refers to NCVX, please cite both
        NCVX and GRANSO paper:

        [1] Buyun Liang, and Ju Sun. 
            NCVX: A User-Friendly and Scalable Package for Nonconvex 
            Optimization in Machine Learning. arXiv preprint arXiv:2111.13984 (2021).
            Available at https://arxiv.org/abs/2111.13984

        [2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton 
            A BFGS-SQP method for nonsmooth, nonconvex, constrained 
            optimization and its evaluation using relative minimization 
            profiles, Optimization Methods and Software, 32(1):148-181, 2017.
            Available at https://dx.doi.org/10.1080/10556788.2016.1208749

        Change Log:
            
            Buyun Dec 20, 2021 (NCVX Version 1.0.0):
                getNvar.py is introduced in NCVX

        For comments/bug reports, please visit the NCVX webpage:
        https://github.com/sun-umn/NCVX
        
        NCVX Version 1.0.0, 2021, see AGPL license info below.

        =========================================================================
        |  NCVX (NonConVeX): A User-Friendly and Scalable Package for           |
        |  Nonconvex Optimization in Machine Learning.                          |
        |                                                                       |
        |  Copyright (C) 2021 Buyun Liang                                       |
        |                                                                       |
        |  This file is part of NCVX.                                           |
        |                                                                       |
        |  NCVX is free software: you can redistribute it and/or modify         |
        |  it under the terms of the GNU Affero General Public License as       |
        |  published by the Free Software Foundation, either version 3 of       |
        |  the License, or (at your option) any later version.                  |
        |                                                                       |
        |  GRANSO is distributed in the hope that it will be useful,            |
        |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
        |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
        |  GNU Affero General Public License for more details.                  |
        |                                                                       |
        |  You should have received a copy of the GNU Affero General Public     |
        |  License along with this program.  If not, see                        |
        |  <http://www.gnu.org/licenses/agpl.html>.                             |
        =========================================================================
    """
    try:
        f.backward(retain_graph=True)
        if double_precision:
            torch_dtype = torch.double
        else:
            torch_dtype = torch.float

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
    return f_grad_vec

def getObjGradDL(nvar,model,f, torch_device, double_precision):
    """
    getObjGradDL:
        getObjGrad obtains gradient of objective function defined by torch.nn module
        by using pytorch autodiff
        
        If you publish work that uses or refers to NCVX, please cite both
        NCVX and GRANSO paper:

        [1] Buyun Liang, and Ju Sun. 
            NCVX: A User-Friendly and Scalable Package for Nonconvex 
            Optimization in Machine Learning. arXiv preprint arXiv:2111.13984 (2021).
            Available at https://arxiv.org/abs/2111.13984

        [2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton 
            A BFGS-SQP method for nonsmooth, nonconvex, constrained 
            optimization and its evaluation using relative minimization 
            profiles, Optimization Methods and Software, 32(1):148-181, 2017.
            Available at https://dx.doi.org/10.1080/10556788.2016.1208749

        Change Log:
            
            Buyun Dec 20, 2021 (NCVX Version 1.0.0):
                getObjGradDL.py is introduced in NCVX

        For comments/bug reports, please visit the NCVX webpage:
        https://github.com/sun-umn/NCVX
        
        NCVX Version 1.0.0, 2021, see AGPL license info below.

        =========================================================================
        |  NCVX (NonConVeX): A User-Friendly and Scalable Package for           |
        |  Nonconvex Optimization in Machine Learning.                          |
        |                                                                       |
        |  Copyright (C) 2021 Buyun Liang                                       |
        |                                                                       |
        |  This file is part of NCVX.                                           |
        |                                                                       |
        |  NCVX is free software: you can redistribute it and/or modify         |
        |  it under the terms of the GNU Affero General Public License as       |
        |  published by the Free Software Foundation, either version 3 of       |
        |  the License, or (at your option) any later version.                  |
        |                                                                       |
        |  GRANSO is distributed in the hope that it will be useful,            |
        |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
        |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
        |  GNU Affero General Public License for more details.                  |
        |                                                                       |
        |  You should have received a copy of the GNU Affero General Public     |
        |  License along with this program.  If not, see                        |
        |  <http://www.gnu.org/licenses/agpl.html>.                             |
        =========================================================================
    """
    f.backward()
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

    return f_grad_vec