import torch
from torch import conj
from pygransoStruct import GeneralStruct

def bfgsHessianInverseLimitedMem(H0,scaleH0,fixed_scaling,nvec,restart_data,device):
#    bfgsHessianInverseLimitedMem:
#        An object that maintains and updates a L-BFGS approximation to the 
#        inverse Hessian.

    H_obj = H_obj_struct(H0,scaleH0,fixed_scaling,nvec,restart_data,device)

    return H_obj


class H_obj_struct:

    def __init__(self,H0,scaleH0,fixed_scaling,nvec,restart_data,device):

        self.H0 = H0
        self.fixed_scaling = fixed_scaling
        self.nvec = nvec
        self.device = device

        n = H0.shape[0]
        self.S = torch.zeros((n,nvec)).to(device=self.device, dtype=torch.double)
        self.Y = torch.zeros((n,nvec)).to(device=self.device, dtype=torch.double)
        self.rho = torch.zeros((1,nvec)).to(device=self.device, dtype=torch.double)
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
            self.cols = restart_data.S.shape[1]
            if self.cols > self.nvec:
                self.S = restart_data.S[:,0:self.nvec]
                self.Y = restart_data.Y[:,0:self.nvec]
                self.rho = restart_data.rho[0:self.nvec]
                self.count = self.nvec
            else:
                self.S[:,0:self.cols] = restart_data.S
                self.Y[:,0:self.cols] = restart_data.Y
                self.rho[0:self.cols] = restart_data.rho
            self.gamma = restart_data.gamma
        
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
            self.rho = torch.hstack((self.rho[:,1:], torch.tensor(rho_new)))
        
        self.updates += 1
        if damped:
            self.damped_updates += 1

        return skipped

    def applyH(self,q_in):
        q = q_in.detach().clone()
        #  q might be a matrix, not just a vector, so we want to apply
        #  multiplication to all columns of q
        self.cols    = q.shape[1]
        alpha   = torch.zeros((self.count,self.cols)).to(device=self.device, dtype=torch.double)
      
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
        data = GeneralStruct()
        setattr(data,'S',self.S[:,0:self.count])
        setattr(data,'Y',self.Y[:,0:self.count])
        setattr(data,'rho',self.rho[0:self.count])
        setattr(data,'gamma',self.gamma) 
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