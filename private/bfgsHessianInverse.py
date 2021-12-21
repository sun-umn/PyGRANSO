import numpy as np
from ncvxStruct import GeneralStruct
import torch

def bfgsHessianInverse(H,scaleH0):
    """
    bfgsHessianInverse:
        An object that maintains and updates a BFGS approximation to the inverse Hessian.

        INPUT:
            H               [ positive definite matrix ]
                Initial inverse Hessian approximation
        
            scaleH0         [ logical ]
                Logical indicating whether or not to employ scaling on the
                first BFGS update.  Scaling is never applied on any subsequent
                update.
        
        OUTPUT:
            H_obj           [ a struct ] 
                A struct containing function handles for maintaining a BFGS
                inverse Hessian approximation.
        
                skipped = H_obj.update(s,y,sty,damped)
                Attempt a BFGS update.
                INPUT:
                    s           BFGS vector: s = x_{k+1} - x_k = alpha@d
                    y           BFGS vector: y = g_{k_1} - g_k
                    sty         s.T@y
                    damped      logical: whether or not damping was applied to
                                y to ensure that sty > 0 
                OUTPUT:
                    skipped     an integer indicating the status:
                                    skipped = 0:    successful BFGS update
                                    skipped = 1:    update but without scaling
                                    skipped = 2:    failed, s.T@*y <= 0
                                    skipped = 3:    failed, contained nans/infs
        
                Hx = H_obj.applyH(x)     
                Multiply matrix/vector x by H, the current BFGS inverse Hessian
                approximation
        
                H = H_obj.getState()
                Return H, the current BFGS inverse Hessian approximation
        
                data = H_obj.getCounts()
                Returns a struct of data with counts of the following:
                    requests            total requests to update
                    updates             total accepted updates
                    damped_requests     request was a damped update
                    damped_updates      update was damped
                    sty_fails           sty <= 0 prevented update
                    infnan_fail         inf/nan in result prevented update
                    scaling_skips       numerical issue prevent scaling 
        
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
            bfgsHessianInverse.m introduced in GRANSO Version 1.0.
            
            Buyun Dec 20, 2021 (NCVX Version 1.0.0):
                bfgsHessianInverse.py is translated from bfgsHessianInverse.m in GRANSO Version 1.6.4. 

        For comments/bug reports, please visit the NCVX webpage:
        https://github.com/sun-umn/NCVX
        
        NCVX Version 1.0.0, 2021, see AGPL license info below.

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
    H_obj = H_obj_struct(H,scaleH0)
    return H_obj

class H_obj_struct:
    
    def __init__(self,H,scaleH0):
        self.requests        = 0
        self.updates         = 0
        self.damped_requests = 0
        self.damped_updates  = 0
        self.scale_fails     = 0
        self.sty_fails       = 0
        self.infnan_fails    = 0
        self.H = H
        self.scaleH0 = scaleH0

    def update(self,s,y,sty,damped=False):
        self.requests += 1
        skipped  = 0
        
        if damped == True:
            self.damped_requests += 1

        #   sty > 0 should hold in theory but it can fail due to rounding
        #  errors, even if BFGS damping is applied. 
        if sty <= 0:
            skipped = 2
            self.sty_fails += 1
            return skipped
        
        # sty > 0 so attempt BFGS update
        if self.scaleH0:
            # for full BFGS, Nocedal and Wright recommend scaling I
            # before the first update only
            
            # gamma = sty/np.dot(np.transpose(y),y) 
            gamma = sty/(torch.conj(y.t()) @ y) 
            gamma = gamma.item()

            if np.isinf(gamma) or np.isnan(gamma):
                skipped     = 1
                self.scale_fails += 1
            else:
                self.H *= gamma
            
            self.scaleH0         = False # only allow first update to be scaled

        rho = (1./sty)[0][0]
        Hy = self.H @ y
        rhoHyst = (rho*Hy) @ torch.conj(s.t())
        
        #  ytHy could be < 0 if H not numerically pos def
        ytHy = torch.conj(y.t()) @ Hy
        sstfactor = max([(rho*rho*ytHy + rho).item(),  0])
        sscaled = np.sqrt(sstfactor)*s
        H_new = self.H - (torch.conj(rhoHyst.t()) + rhoHyst) + sscaled @ torch.conj(sscaled.t())
        H_vec = torch.reshape(H_new, (torch.numel(H_new),1))
        notInf_flag = torch.all(torch.isinf(H_vec) == False)
        notNan_flag = torch.all(torch.isnan(H_vec) == False)

        if notInf_flag and notNan_flag:
            self.H = H_new
            self.updates += 1
            if damped: 
                self.damped_updates += 1
        else:
            skipped = 3
            self.infnan_fails += 1

        return skipped

    def applyH(self,q):
        r = self.H @q 
        return r

    def getState(self):
        H_out = self.H
        return H_out

    def getCounts(self):
        counts = GeneralStruct()
        setattr(counts,"requests",self.requests)
        setattr(counts,"updates",self.updates)
        setattr(counts,"damped_requests",self.damped_requests)
        setattr(counts,"damped_updates",self.damped_updates)
        setattr(counts,"scaling_skips",self.scale_fails)
        setattr(counts,"sty_fails",self.sty_fails)
        setattr(counts,"infnan_fails",self.infnan_fails)
        return counts


