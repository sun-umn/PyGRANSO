from private.isRealValued import isRealValued
from private.isARealNumber import isARealNumber

def isRestartData(data):
    """  
    isRestartData:
        Checks whether data is lbfgs restart data.
        
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
                isString.py is introduced in NCVX

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

    if isinstance(data,dict) == False:
      return False

    if  'S' in data and 'Y' in data and 'rho' in data and 'gamma' in data:
      return True
    
    return False

      # if  isARealNumber(data['gamma']):
      #   try:
      #     [n_S,cols_S]    = data['S'].shape
      #     [n_Y,cols_Y]    = data['Y'].shape
      #     [_,cols_rho]    = data['rho'].shape
      #     return True
      #   except Exception as e:
      #     return False

    