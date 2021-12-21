import torch
from torch import conj
from ncvxStruct import GeneralStruct

def bfgsHessianInverseLimitedMem(H0,scaleH0,fixed_scaling,nvec,restart_data,device, double_precision):
    """
    bfgsHessianInverseLimitedMem:
        An object that maintains and updates a L-BFGS approximation to the 
        inverse Hessian.
    
        INPUT:
            HO                  [ positive definite matrix ]
                Initial inverse Hessian approximation to use in L-BFGS update
                formula.
        
            scaleH0             [ logical ]
                Logical indicating whether or not to employ scaling to H0.
        
            fixed_scaling       [ logical ]
                When fixed_scaling is true (and scaleH0 is also true), the
                first calculated value of the scaling parameter will be used
                for all subsequent updates.  Otherwise, a new value of the
                scaling parameter will be calculated on each update.
        
            nvec                [ positive integer ]
                number of vectors to store for L-BFGS.
        
            restart_data        [ struct ]
                Struct of data to warm-start L-BFGS from previous information:
                .S          n by m real-valued finite matrix of s vectors
                .Y          n by m real-valued finite matrix of y vectors
                .rho        length m real-valued row vector of 1/sty values
                .gamma      real finite value
                H_obj.getState() returns a struct containing this data.
                        
        OUTPUT:
            H_obj               [ a struct ] 
                A struct containing function handles for maintaining an L-BFGS
                inverse Hessian approximation.
        
                skipped = H_obj.update(s,y,sty,damped)
                Attempt an L-BFGS update.
                INPUT:
                    s           BFGS vector: s = x_{k+1} - x_k = alpha@d
                    y           BFGS vector: y = g_{k_1} - g_k
                    sty         s.T@y
                    damped      logical: whether or not damping was applied to
                                y to ensure that sty > 0 
                OUTPUT:
                    skipped     an integer indicating the status:
                                    skipped = 0:    successful L-BFGS update
                                    skipped = 1:    update but without scaling
                                    skipped = 2:    failed, s.T@y <= 0
                                    skipped = 3:    failed, contained nans/infs
        
                Hx = H_obj.applyH(x) 
                Multiply matrix/vector x by H, the current L-BFGS inverse
                Hessian approximation
        
                H = H_obj.getState()
                Returns a struct with current L-BFGS state:
                    .S          n by m real-valued finite matrix of s vectors
                    .Y          n by m real-valued finite matrix of y vectors
                    .rho        length m real-valued row vector of 1/sty values
                    .gamma      real finite value
                        
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
            bfgsHessianInverseLimitedMem.m introduced in GRANSO Version 1.5
            
            Buyun Dec 20, 2021 (NCVX Version 1.0.0):
                bfgsHessianInverseLimitedMem.py is translated from bfgsHessianInverseLimitedMem.m in GRANSO Version 1.6.4. 

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
    H_obj = H_obj_struct(H0,scaleH0,fixed_scaling,nvec,restart_data,device,double_precision)
    return H_obj


class H_obj_struct:

    def __init__(self,H0,scaleH0,fixed_scaling,nvec,restart_data,device,double_precision):

        self.H0 = H0
        self.fixed_scaling = fixed_scaling
        self.nvec = nvec
        self.device = device
        self.double_precision = double_precision

        n = H0.shape[0]
        if double_precision:
            self.torch_dtype = torch.double
        else:
            self.torch_dtype = torch.float

        self.S = torch.zeros((n,nvec)).to(device=self.device, dtype=self.torch_dtype)
        self.Y = torch.zeros((n,nvec)).to(device=self.device, dtype=self.torch_dtype)
        self.rho = torch.zeros((1,nvec)).to(device=self.device, dtype=self.torch_dtype)

        self.gamma = 1
        self.count = 0
        self.update_gamma = scaleH0

        self.requests = 0
        self.updates = 0
        self.damped_requests = 0
        self.damped_updates = 0
        self.scale_fails = 0
        self.sty_fails = 0
        self.infnan_fails = 0

        if restart_data != None:
            self.cols = restart_data['S'].shape[1]
            if self.cols > self.nvec:
                self.S = restart_data['S'][:,0:self.nvec]
                self.Y = restart_data['Y'][:,0:self.nvec]
                self.rho = restart_data['rho'][0:self.nvec]
                self.count = self.nvec
            else:
                self.S[:,0:self.cols] = restart_data['S']
                self.Y[:,0:self.cols] = restart_data['Y']
                self.rho[0:self.cols] = restart_data['rho']
                self.count = self.cols
            self.gamma = restart_data['gamma']
        
    def update(self,s,y,sty,damped = False):
  
        self.requests += 1
        skipped = 0
        
        if damped:
            self.damped_requests += 1
        
        #   sty > 0 should hold in theory but it can fail due to rounding
        #   errors, even with if BFGS damping is applied. 
        if sty <= 0:
            skipped = 2
            self.sty_fails += 1
            return skipped
        
        #  We should also check that s and y only have finite entries.
        if  torch.any(torch.isinf(s)) or torch.any(torch.isnan(s)) or torch.any(torch.isinf(y)) or torch.any(torch.isnan(y)): 
            skipped = 3
            self.infnan_fails += 1
            return skipped
        
        rho_new = 1/(sty);  # this will be finite since sty > 0
        
        if self.update_gamma: 
            gamma_new       = 1 / (rho_new * (y.T @ y))
            if torch.isinf(gamma_new) or torch.isnan(gamma_new):
                #  can still apply update with scaling disabled and/or
                #  previous value of scaling parameter.  We'll update the
                #  latter approach.
                skipped = 1
                self.scale_fails += 1
            else:
                self.gamma = gamma_new       
                if self.fixed_scaling:
                    self.update_gamma = False
                
        
        if self.count < self.nvec:
            
            self.S[:,self.count] = s[:,0]
            self.Y[:,self.count] = y[:,0]
            self.rho[0,self.count] = rho_new    
            self.count += 1
        else:
            self.S = torch.hstack((self.S[:,1:], s)) 
            self.Y = torch.hstack((self.Y[:,1:], y)) 
            self.rho = torch.hstack((self.rho[:,1:], rho_new))
        
        self.updates += 1
        if damped:
            self.damped_updates += 1

        return skipped

    def applyH(self,q_in):
        q = q_in.detach().clone()
        #  q might be a matrix, not just a vector, so we want to apply
        #  multiplication to all columns of q
        self.cols    = q.shape[1]
        alpha = torch.zeros((self.count,self.cols)).to(device=self.device, dtype=self.torch_dtype)

        #  Now apply first updates to the columns of q
        for j in reversed(range(self.count)):
            alpha[j,:]  = self.rho[0,j] * (self.S[:,j].T  @ q)
            y = self.Y[:,j]
            for k in range(self.cols):
                q[:,k]  = q[:,k] - alpha[j,k] * y
        
        #  Apply the sparse matvec and (possible) scaling 
        r = self.gamma*(self.H0@q)
        
        #  Finally apply updates to the columns of r
        for j in range(self.count):
            beta = self.rho[0,j] * ( conj(self.Y[:,j]).T @ r)
            s = self.S[:,j]
            for k in range(self.cols):
                r[:,k] = r[:,k] + (alpha[j,k] - beta[k]) * s
        return r

    def getState(self):
        data = {'S':self.S[:,0:self.count], 'Y':self.Y[:,0:self.count], 'rho':self.rho[0:self.count], 'gamma':self.gamma}
        return data

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