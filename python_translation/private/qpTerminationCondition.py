import numpy as np
from private.solveQP import Class_solveQP

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

        mu          = penaltyfn_at_x.mu
        l           = gradient_samples.size
        p           = l * len(penaltyfn_at_x.ci)
        q           = l * len(penaltyfn_at_x.ce)                                
    
        F           = penaltyfn_at_x.f * np.ones((l,1)) 
        
        CI = penaltyfn_at_x.ci
        CE = penaltyfn_at_x.ce
        for i in range(l-1):
            CI_new          = np.vstack(CI,CI)
            CE_new          = np.vstack(CE,CE)
            CI = CI_new
            CE = CE_new
        
        # convert cell array fields F, CI, CE to struct array with same
        grads_array = gradient_samples[0]
        #  convert struct array into individual arrays of samples
        F_grads     = grads_array.F      # n by l
        CI_grads    = grads_array.CI     # n by p
        CE_grads    = grads_array.CE     # n by q 
    
        #  Set up arguments for quadprog interface
        self.all_grads   = np.hstack((CE_grads, F_grads, CI_grads))
        Hinv_grads  = apply_Hinv(self.all_grads)
        self.H           = self.all_grads.T @ Hinv_grads
        #  Fix H since numerically, it is unlikely to be _perfectly_ symmetric 
        self.H           = (self.H + self.H.T) / 2
        f           = -np.vstack((CE, F, CI))
        LB          = np.vstack((-np.ones((q,1)), np.zeros((l+p,1))))
        UB          = np.vstack((np.ones((q,1)), mu*np.ones((l,1)), np.ones((p,1))))  
        Aeq         = np.hstack((np.zeros((1,q)), np.ones((1,l)), np.zeros((1,p))))  
        beq         = mu

        # Choose solver
        if QPsolver == "gurobi":
            solveQP_obj = Class_solveQP()
            #  formulation of QP has no 1/2
            self.solveQP_fn = lambda H: solveQP_obj.solveQP(H,f,Aeq,beq,LB,UB,QPsolver)

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
        stat_type   = 1
        x = self.solveQP_fn(self.H)
        # return [x,lambdas,stat_type,ME]
           
        print("qpTerminationCondition: ignore other 3 Fall back strategies for now")
        
        return [x,lambdas,stat_type,ME]
    