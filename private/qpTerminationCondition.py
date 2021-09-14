import numpy as np
import torch
from private.solveQP import solveQP
from dbg_print import dbg_print
from numpy import conjugate as conj
from numpy import linalg as LA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

class qpTC:
    def __init__(self):
        pass
    def qpTerminationCondition(self,penaltyfn_at_x, gradient_samples, apply_Hinv, QPsolver):
        """
        qpTerminationCondition:
        computes the smallest vector in the convex hull of gradient samples
        provided in cell array gradient samples, given the inverse Hessian
        (or approximation to it)
        """
        # obtain # of variables
        n = torch.numel(gradient_samples[0].F)
        mu          = penaltyfn_at_x.mu
        l           = gradient_samples.size
        p           = l * len(penaltyfn_at_x.ci)
        q           = l * len(penaltyfn_at_x.ce)                                
    
        # # debug here:
        # l = 1

        F           = penaltyfn_at_x.f * torch.ones((l,1),device=device, dtype=torch.double) 
        
        CI = penaltyfn_at_x.ci
        CE = penaltyfn_at_x.ce
        # for i in range(l-1):
        #     CI_new          = torch.vstack((CI_new,CI))
        #     CE_new          = torch.vstack((CE_new,CE))
        # if l > 1:
        #     CI = CI_new
        #     CE = CE_new
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
        # n = int(F_grads_tmp.size/l)
        # F_grads = F_grads_tmp.reshape((n,l))
        CI_grads = torch.cat(CI_grads_lst,1) 
        CE_grads = torch.cat(CE_grads_lst,1) 

        dbg_print("check qpTerminationCondition dimension here. line 50")

        #  Set up arguments for quadprog interface
        self.all_grads   = torch.hstack((CE_grads, F_grads, CI_grads))
        Hinv_grads  = apply_Hinv(self.all_grads)
        self.H           = torch.conj(self.all_grads.t()) @ Hinv_grads
        #  Fix H since numerically, it is unlikely to be _perfectly_ symmetric 
        self.H           = (self.H + torch.conj(self.H.t())) / 2
        f           = -torch.vstack((CE, F, CI))
        LB          = torch.vstack( (-torch.ones((q,1),device=device, dtype=torch.double), torch.zeros((l+p,1),device=device, dtype=torch.double)) ) 
        UB          = torch.vstack((torch.ones((q,1),device=device, dtype=torch.double), mu*torch.ones((l,1),device=device, dtype=torch.double), torch.ones((p,1),device=device, dtype=torch.double))  ) 
        Aeq         = torch.hstack((torch.zeros((1,q),device=device, dtype=torch.double), torch.ones((1,l),device=device, dtype=torch.double), torch.zeros((1,p),device=device, dtype=torch.double)) ) 
        beq         = mu

        # Choose solver
        if QPsolver == "gurobi":
            #  formulation of QP has no 1/2
            self.solveQP_fn = lambda H: solveQP(H,f,Aeq,beq,LB,UB,QPsolver)
        elif QPsolver == "osqp":
            #  formulation of QP has no 1/2
            self.solveQP_fn = lambda H: solveQP(H,f,Aeq,beq,LB,UB,QPsolver)

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
            print(e)
            print("PyGRANSO:qpTerminationCondition type 1 failure")
           

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
            print(e)
            print("PyGRANSO:qpTerminationCondition type 2 failure")
    
        dbg_print("ignore revert to MATLAB quadprog")

        # % Fall back strategy #2: revert to MATLAB's quadprog, if user is
        # % using a different quadprog solver and reattempt with original H
        # if ~isDefaultQuadprog() 
        #     users_paths = path;
        #     % put MATLAB's quadprog on top of path so that it is called
        #     addpath(getDefaultQuadprogPath());
        #     try 
        #         stat_type   = 3;
        #         [x,lambdas] = solveQP_fn(H);  
        #     catch err
        #         ME = ME.addCause(err);
        #     end
        #     % restore user's original list of paths (and their order!)
        #     path(users_paths); 
            
        #     % solve succeeded
        #     if ~isempty(x)
        #         return
        #     end
        # end 

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
            print(e)
            print("PyGRANSO:qpTerminationCondition type 4 failure")
