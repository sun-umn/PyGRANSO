import torch

def getCiVec(ci,torch_device, double_precision):
    """
    getCiVec:
        getCiVec transforms the original tensor form constrained function into vector form

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
                getCiVec.py is introduced in NCVX

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
    #  number of constraints
    nconstr = 0
    # get # of constraints
    # current constraint, e.g., c1, c2
    for constr_i in ci.__dict__.keys():
        constrMatrix = getattr(ci,constr_i)
        nconstr = nconstr + torch.numel(constrMatrix)

    if double_precision:
        torch_dtype = torch.double
    else:
        torch_dtype = torch.float

    # inquality constraints
    ci_vec_torch = torch.zeros((nconstr,1),device=torch_device, dtype=torch_dtype)
   
   
    curIdx = 0
    # nconstr_ci = genral_struct()
    nconstr_ci_total = 0
    # current constraint, e.g., c1, c2
    for constr_i in ci.__dict__.keys():
        constrMatrix = getattr(ci,constr_i)
        ci_vec_torch[curIdx:curIdx + torch.numel(constrMatrix)] = torch.reshape(constrMatrix,(torch.numel(constrMatrix),1))
        curIdx = curIdx + torch.numel(constrMatrix)
        # setattr(nconstr_ci,constr_i,torch.numel(constrMatrix))
        nconstr_ci_total += torch.numel(constrMatrix)

    ci_vec = ci_vec_torch.detach() # detach from current computational graph

    return [ci_vec,ci_vec_torch,nconstr_ci_total]