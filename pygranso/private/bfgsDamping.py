from numpy import conjugate as conj
import torch

def bfgsDamping(damping,applyH,s,y,sty):
    """
    bfgsDamping:
        This function implements Procedure 18.2 from Nocedal and Wright,
        which ensures that the BFGS update is always well defined.

        INPUT:
            damping         [ real number in [0,1] ]
                constant to determine how frequently and aggressively damping
                will be applied.

            applyH          [ function handle ]
                Returns H@x, where H is the BFGS inverse Hessian approximation,
                for single input argument x

            s               [ real finite column vector ]
                BFGS vector: s = x_{k+1} - x_k = alpha@d

            y               [ real finite column vector ]
                BFGS vector: y = g - gprev

            sty             [ real finite scalar ]
                sty = s.T@y

        OUTPUT:
            y               [ real finite column vector ]
                possibly updated version of y

            sty             [ real finite scalar ]
                possibly updated version of sty

            damped          [ logical ]
                true if damping was applied

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

        bfgsDamping.py (introduced in PyGRANSO v1.0.0)
        Copyright (C) 2016-2021 Tim Mitchell

        This is a direct port of bfgsDamping.m from GRANSO v1.6.4.
        Ported from MATLAB to Python by Buyun Liang 2021.

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

    damped      = False
    Hs          = applyH(s)
    stHs        = torch.conj(s.t())@Hs

    if sty < damping * stHs:
        theta   = ((1 - damping) * stHs) / (stHs - sty)
        y       = theta * y + (1 - theta) * Hs
        sty     = theta * sty + (1- theta) * stHs # s.T@y;
        damped  = True

    return [y,sty,damped]
