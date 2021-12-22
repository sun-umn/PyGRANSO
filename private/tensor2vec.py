import numpy as np
import torch
from private.getObjGrad import getObjGradDL,getObjGrad
from private.vec2tensor import vec2tensor
from private.getCiVec import getCiVec
from private.getCiGradVec import getCiGradVec

def obj_eval(eval_obj, x, var_dim_map, data_in = None):
    """
    obj_eval:
        obj_eval makes an objective evaluation function used for backtrack line search
        
        If you publish work that uses or refers to PyGRANSO, please cite both
        PyGRANSO and GRANSO paper:

        [1] Buyun Liang, and Ju Sun. 
            PyGRANSO: A User-Friendly and Scalable Package for Nonconvex 
            Optimization in Machine Learning. arXiv preprint arXiv:2111.13984 (2021).
            Available at https://arxiv.org/abs/2111.13984

        [2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton 
            A BFGS-SQP method for nonsmooth, nonconvex, constrained 
            optimization and its evaluation using relative minimization 
            profiles, Optimization Methods and Software, 32(1):148-181, 2017.
            Available at https://dx.doi.org/10.1080/10556788.2016.1208749

        Change Log:
            
            Buyun Dec 20, 2021 (PyGRANSO Version 1.0.0):
                obj_eval.py is introduced in PyGRANSO

        For comments/bug reports, please visit the PyGRANSO webpage:
        https://github.com/sun-umn/PyGRANSO
        
        PyGRANSO Version 1.0.0, 2021, see AGPL license info below.

        =========================================================================
        |  PyGRANSO: A User-Friendly and Scalable Package for                   |
        |  Nonconvex Optimization in Machine Learning.                          |
        |                                                                       |
        |  Copyright (C) 2021 Buyun Liang                                       |
        |                                                                       |
        |  This file is part of PyGRANSO.                                       |
        |                                                                       |
        |  PyGRANSO is free software: you can redistribute it and/or modify     |
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
    X_struct = vec2tensor(x,var_dim_map)
    if data_in == None:
        f = eval_obj(X_struct)
    else:
        f = eval_obj(X_struct,data_in)
    return f

def tensor2vec(combinedFunction,x,var_dim_map,nvar,data_in = None,  torch_device = torch.device('cpu'), model = None, double_precision=True):
    """
    tensor2vec
        Return vector form objective and constraints information required by PyGRANSO

        If you publish work that uses or refers to PyGRANSO, please cite both
        PyGRANSO and GRANSO paper:

        [1] Buyun Liang, and Ju Sun. 
            PyGRANSO: A User-Friendly and Scalable Package for Nonconvex 
            Optimization in Machine Learning. arXiv preprint arXiv:2111.13984 (2021).
            Available at https://arxiv.org/abs/2111.13984

        [2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton 
            A BFGS-SQP method for nonsmooth, nonconvex, constrained 
            optimization and its evaluation using relative minimization 
            profiles, Optimization Methods and Software, 32(1):148-181, 2017.
            Available at https://dx.doi.org/10.1080/10556788.2016.1208749

        Change Log:
            
            Buyun Dec 20, 2021 (PyGRANSO Version 1.0.0):
                tensor2vec.py is introduced in PyGRANSO

        For comments/bug reports, please visit the PyGRANSO webpage:
        https://github.com/sun-umn/PyGRANSO
        
        PyGRANSO Version 1.0.0, 2021, see AGPL license info below.

        =========================================================================
        |  PyGRANSO: A User-Friendly and Scalable Package for                   |
        |  Nonconvex Optimization in Machine Learning.                          |
        |                                                                       |
        |  Copyright (C) 2021 Buyun Liang                                       |
        |                                                                       |
        |  This file is part of PyGRANSO.                                       |
        |                                                                       |
        |  PyGRANSO is free software: you can redistribute it and/or modify     |
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


