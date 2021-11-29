import numpy as np
import torch
from private.solveQP import solveQP
from numpy import conjugate as conj
from numpy import linalg as LA
import traceback,sys

class qpTC:
    def __init__(self):
        pass
    def qpTerminationCondition(self,penaltyfn_at_x, gradient_samples, apply_Hinv, QPsolver, torch_device, double_precision):
        """
        qpTerminationCondition:
        computes the smallest vector in the convex hull of gradient samples
        provided in cell array gradient samples, given the inverse Hessian
        (or approximation to it)
        """
        if double_precision:
            torch_dtype = torch.double
        else:
            torch_dtype = torch.float

        # obtain # of variables
        n = torch.numel(gradient_samples[0].F)
        mu          = penaltyfn_at_x.mu
        l           = gradient_samples.size
        p           = l * len(penaltyfn_at_x.ci)
        q           = l * len(penaltyfn_at_x.ce)                                
    

        F           = penaltyfn_at_x.f * torch.ones((l,1),device=torch_device, dtype=torch_dtype) 
        
        CI = penaltyfn_at_x.ci
        CE = penaltyfn_at_x.ce

        CI = CI.repeat(l,1)
        CE = CE.repeat(l,1)
        
        
        F_grads_lst = []
        CI_grads_lst = []
        CE_grads_lst = []
        
        for i in range(l):
            # convert cell array fields F, CI, CE to struct array with same
            grads_array = gradient_samples[i]
            #  convert struct array into individual arrays of samples
            F_grads_lst.append(grads_array.F)     # n by l
            CI_grads_lst.append(grads_array.CI)     # n by p
            CE_grads_lst.append(grads_array.CE)     # n by q
        F_grads = torch.cat(F_grads_lst,1) 
        CI_grads = torch.cat(CI_grads_lst,1) 
        CE_grads = torch.cat(CE_grads_lst,1) 

        #  Set up arguments for quadprog interface
        self.all_grads   = torch.hstack((CE_grads, F_grads, CI_grads))
        Hinv_grads  = apply_Hinv(self.all_grads)
        self.H           = torch.conj(self.all_grads.t()) @ Hinv_grads
        #  Fix H since numerically, it is unlikely to be _perfectly_ symmetric 
        self.H           = (self.H + torch.conj(self.H.t())) / 2
        f           = -torch.vstack((CE, F, CI))
        LB          = torch.vstack( (-torch.ones((q,1),device=torch_device, dtype=torch_dtype), torch.zeros((l+p,1),device=torch_device, dtype=torch_dtype)) ) 
        UB          = torch.vstack((torch.ones((q,1),device=torch_device, dtype=torch_dtype), mu*torch.ones((l,1),device=torch_device, dtype=torch_dtype), torch.ones((p,1),device=torch_device, dtype=torch_dtype))  ) 
        Aeq         = torch.hstack((torch.zeros((1,q),device=torch_device, dtype=torch_dtype), torch.ones((1,l),device=torch_device, dtype=torch_dtype), torch.zeros((1,p),device=torch_device, dtype=torch_dtype)) ) 
        beq         = mu

        # Choose solver
        if QPsolver == "gurobi":
            #  formulation of QP has no 1/2
            self.solveQP_fn = lambda H: solveQP(H,f,Aeq,beq,LB,UB,QPsolver,torch_device, double_precision)
        elif QPsolver == "osqp":
            #  formulation of QP has no 1/2
            self.solveQP_fn = lambda H: solveQP(H,f,Aeq,beq,LB,UB,QPsolver,torch_device, double_precision)

        [y,_,qps_solved,ME] = self.solveQPRobust()
      
        #  If the QP solve(s) failed, return infinite vector so it can't
        #  possibly trigger BFGS-SQP's convergence criteria
        if y.size == 0:
            #  its length is equal to the number of variables
            d = np.inf * np.ones((Hinv_grads.shape[0],1)); 
        else:
            d = -Hinv_grads @ y

        return [d,qps_solved,ME]

    def solveQPRobust(self):
       
        x       = None
        lambdas = None # not used here
        ME      = None # ignore other 3 Fall back strategies for now
        
        #  Attempt to solve QP
        try:
            stat_type   = 1
            x = self.solveQP_fn(self.H)
            return [x,lambdas,stat_type,ME]
        except Exception as e:
            print("NCVX:qpTerminationCondition type 1 failure")
            print(traceback.format_exc())
           

        #  QP solver failed, possibly because H was numerically nonconvex,
        #  i.e. H may have tiny negative eigenvalues close to zero because
        #  of rounding errors
       
        #  Fall back strategy #1: replace Hinv with identity and try again
        try:
            stat_type = 2
            R = conj(self.all_grads.T) @ self.all_grads
            R = (R + conj(R.T))/2
            x = self.solveQP_fn(R)
            return [x,lambdas,stat_type,ME]
        except Exception as e:
            print("NCVX:qpTerminationCondition type 2 failure")
            print(traceback.format_exc())
    
        # % Fall back strategy #2: revert to MATLAB's quadprog, if user is
        # % using a different quadprog solver and reattempt with original H
        # Skip in NCVX

        #  Fall back strategy #3: regularize H - this could be expensive
        #  Even though min(eig(Hreg)) may still be tiny negative number,
        #  this mild regularization seems to often be sufficient prevent 
        #  MOSEK from complaining about nonconvexity and aborting. 

        try:
            stat_type = 4
            [D,V] = LA.eig(self.H)
            dvec = [x if x >= 0 else 0 for x in D]
            Hreg = conj(V.T) @ np.diag(dvec) @ conj(V.T)
            x = self.solveQP_fn(Hreg)
            return [x,lambdas,stat_type,ME]
        except Exception as e:
            print("NCVX:qpTerminationCondition type 4 failure")
            print(traceback.format_exc())
