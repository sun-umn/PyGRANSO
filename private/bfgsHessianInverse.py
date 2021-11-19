import numpy as np
from ncvxStruct import GeneralStruct
import torch

def bfgsHessianInverse(H,scaleH0):
    """
    bfgsHessianInverse:
        An object that maintains and updates a BFGS approximation to the inverse Hessian.
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


