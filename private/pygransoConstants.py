def pygransoConstants():
    """
    pygransoConstants:
        Simple routine for defining constants for use in:
            1) pygranso 
            2) pygransoOptions 
            3) pygransoOptionsAdvanced

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
                gransoConstants.m introduced in GRANSO Version 1.0.

                Buyun Dec 20, 2021 (PyGRANSO Version 1.0.0):
                    pygransoConstants.py is translated from gransoConstants.m in GRANSO Version 1.6.4. 

            For comments/bug reports, please visit the PyGRANSO webpage:
            https://github.com/sun-umn/PyGRANSO
                
            PyGRANSO Version 1.0.0, 2021, see AGPL license info below.

            =========================================================================
            |  GRANSO: GRadient-based Algorithm for Non-Smooth Optimization         |
            |  Copyright (C) 2016 Tim Mitchell                                      |
            |                                                                       |
            |  This file is translated from GRANSO.                                 |
            |                                                                       |
            |  GRANSO is free software: you can redistribute it and/or modify       |
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
    # Number of first fallback level after QP approaches have failed
    POSTQP_FALLBACK_LEVEL       = 2
    # Number of last fallback level (randomly generated search directions)
    LAST_FALLBACK_LEVEL         = 4

    return [POSTQP_FALLBACK_LEVEL, LAST_FALLBACK_LEVEL]