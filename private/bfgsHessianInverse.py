import numpy as np
from numpy.core.numeric import Inf
from pygransoStruct import general_struct
from numpy import conjugate as conj
from dbg_print import dbg_print_1
import torch

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

    # @profile
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
            gamma = sty/(conj(y.T) @ y) 

            if np.isinf(gamma) or np.isnan(gamma):
                skipped     = 1
                self.scale_fails += 1
            else:
                self.H *= gamma
            
            self.scaleH0         = False # only allow first update to be scaled
        

        dbg_print_1("start torch accelaeration")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # sty_gpu = torch.from_numpy(sty).to(device=device)
        H_gpu = torch.from_numpy(self.H).to(device=device)
        s_gpu = torch.from_numpy(s).to(device=device)
        # y_gpu = torch.from_numpy(y).to(device=device)
        s_gpu = torch.from_numpy(s).to(device=device)

        # for formula, see Nocedal and Wright's book
        # M = I - rho*s*y', H = M*H*M' + rho*s*s', so we have
        # H = H - rho*s*y'*H - rho*H*y*s' + rho^2*s*y'*H*y*s' + rho*s*s'
        #  note that the last two terms combine: (rho^2*y'Hy + rho)ss'
        rho = (1./sty)[0][0]

        # Hy = np.dot(self.H,y)
        # rhoHyst = np.dot((rho*Hy),np.transpose(s) ) 
        Hy = self.H @ y
        Hy_gpu = torch.from_numpy(Hy).to(device=device)
        
        

        # rhoHyst = (rho*Hy) @ conj(s.T)
        rhoHyst = (rho*Hy_gpu) @ torch.conj(s_gpu.t())

        #   old version: update may not be symmetric because of rounding
        #  H = H - rhoHyst' - rhoHyst + rho*s*(y'*rhoHyst) + rho*s*s';
        #  new in HANSO version 2.02: make H explicitly symmetric
        #  also saves one outer product
        #  in practice, makes little difference, except H=H' exactly
        
        #  ytHy could be < 0 if H not numerically pos def
        
        # ytHy = np.dot(np.transpose(y),Hy)
        ytHy = conj(y.T) @ Hy
        sstfactor = max([rho*rho*ytHy + rho,  0])[0][0]
        # sscaled = np.sqrt(sstfactor)*s
        sscaled = np.sqrt(sstfactor)*s_gpu
        # H_new = self.H - (conj(rhoHyst.T) + rhoHyst) + sscaled @ conj(sscaled.T)
        H_new = H_gpu - (torch.conj(rhoHyst.t()) + rhoHyst) + sscaled @ torch.conj(sscaled.t())
        # H_new = H_new.cpu().numpy()

        #  only update H if H_new doesn't contain any infs or nans
        # H_vec = np.reshape(H_new, (H_new.size,1))
        # if np.all( (np.logical_or(np.isinf(H_vec),np.isnan(H_vec))) == False ): 
        
        H_vec = torch.reshape(H_new, (torch.numel(H_new),1))
        notInf_flag = torch.all(torch.isinf(H_vec) == False)
        notNan_flag = torch.all(torch.isnan(H_vec) == False)
        dbg_print_1("notInf_flag = {}".format( notInf_flag ) )
        dbg_print_1("notNan_flag = {}".format( notNan_flag ) )
        # if torch.all( (torch.logical_or(torch.isfinite(H_vec),torch.isnan(H_vec))) == False ): 
        if notInf_flag and notNan_flag:
            self.H = H_new.cpu().numpy()
            self.updates += 1
            if damped: 
                self.damped_updates += 1
        else:
            skipped = 3
            self.infnan_fails += 1
        
        #  An aside, for an alternative way of doing the BFGS update:
        #  Add the update terms together first: does not seem to make
        #  significant difference
        #    update = sscaled*sscaled' - (rhoHyst' + rhoHyst);
        #    H = H + update;

        return skipped

    def applyH(self,q):
        r = self.H @q 
        # dbg_print_1("Second-derivative test:")
        # pos_definite_flag = np.sum(np.linalg.eigvals(self.H) > 0)
        # dbg_print_1(pos_definite_flag)
        # dbg_print_1(pos_definite_flag/self.H.shape[0])

        return r

    def getState(self):
        H_out = self.H
        return H_out

    def getCounts(self):
        counts = general_struct()
        setattr(counts,"requests",self.requests)
        setattr(counts,"updates",self.updates)
        setattr(counts,"damped_requests",self.damped_requests)
        setattr(counts,"damped_updates",self.damped_updates)
        setattr(counts,"scaling_skips",self.scale_fails)
        setattr(counts,"sty_fails",self.sty_fails)
        setattr(counts,"infnan_fails",self.infnan_fails)
        return counts


def bfgsHessianInverse(H,scaleH0):
#    bfgsHessianInverse:
#        An object that maintains and updates a BFGS approximation to the 
#        inverse Hessian.

    H_obj = H_obj_struct(H,scaleH0)

    return H_obj