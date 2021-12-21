import numpy as np
import torch

def getCiGradVec(nvar,nconstr_ci_total,var_dim_map,X,ci_vec_torch, torch_device, double_precision):
    """
    getCiGradVec:
        getCiGradVec obtains gradient of constraints function (in the vector form) by using pytorch autodiff

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
                getCiGradVec.py is introduced in NCVX

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
    # gradient of inquality constraints
    if double_precision:
        torch_dtype = torch.double
    else:
        torch_dtype = torch.float

    ci_grad_vec = torch.zeros((nvar,nconstr_ci_total),device=torch_device, dtype=torch_dtype)
   

    for i in range(nconstr_ci_total):
        ci_vec_torch[i].backward(retain_graph=True)
        # current variable, e.g., U
        curIdx = 0
        for var in var_dim_map.keys():
            # corresponding dimension of the variable, e.g, 3 by 2
            dim = var_dim_map.get(var) 
            ci_grad_tmp = getattr(X,var).grad
            if ci_grad_tmp != None:
                varLen = np.prod(dim)
                ci_grad_reshape = torch.reshape(ci_grad_tmp,(varLen,1))[:,0]
                ci_grad_vec[curIdx:curIdx+varLen,i] = ci_grad_reshape 
                curIdx += varLen
                getattr(X,var).grad.zero_()

    return ci_grad_vec