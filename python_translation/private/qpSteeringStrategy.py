from math import inf
import numpy as np
import numpy.linalg as LA
from private.solveQP import solveQP
from dbg_print import dbg_print
from numpy import conjugate as conj

class qpSS:
    def __init__(self):
        pass

    def qpSteeringStrategy( self,penaltyfn_at_x, apply_Hinv, l1_model, ineq_margin, maxit, c_viol, c_mu, QPsolver):
        """
        qpSteeringStrategy:
        attempts to find a search direction which promotes progress towards
        feasibility.  
        """
        self.QPsolver = QPsolver
        mu                  = penaltyfn_at_x.mu
        f_grad              = penaltyfn_at_x.f_grad
        self.ineq                = penaltyfn_at_x.ci
        self.ineq_grad           = penaltyfn_at_x.ci_grad
        self.eq                  = penaltyfn_at_x.ce
        self.eq_grad             = penaltyfn_at_x.ce_grad
        if l1_model:
            predictedViolFn = lambda d: self.predictedViolationReductionL1(d)
            self.violation       = penaltyfn_at_x.tv_l1
        else:
            predictedViolFn = lambda d: self.predictedViolationReduction(d)
            self.violation       = penaltyfn_at_x.tv
        
        
        self.n_ineq              = len(self.ineq)
        n_eq                = len(self.eq)
        
        self.Hinv_f_grad         = apply_Hinv(f_grad)
        
        #  factor to allow for some inaccuracy in the QP solver
        violation_tol       = np.sqrt(np.finfo(np.float64).eps)*max(self.violation,1)
     
         
        #  Set up arguments for quadprog interface
        #  Update (Buyun): Set up arguments for QPALM interface
        self.c_grads             = np.hstack((self.eq_grad, self.ineq_grad))
        self.Hinv_c_grads        = apply_Hinv(self.c_grads)
        self.H                   = conj(self.c_grads.T) @ self.Hinv_c_grads
        #  Fix H since numerically, it is unlikely to be _perfectly_ symmetric 
        self.H                   = (self.H + conj(self.H.T)) / 2
        self.mu_Hinv_f_grad      = mu * self.Hinv_f_grad
        self.f                   = conj(self.c_grads.T) @ self.mu_Hinv_f_grad - np.vstack((self.eq, self.ineq)) 
        self.LB                  = np.vstack((-np.ones((n_eq,1)), np.zeros((self.n_ineq,1))  ))  
        self.UB                  = np.ones((n_eq + self.n_ineq, 1))
        
        #  Identity matrix: compatible with the constraint form in QPALM
        self.A                   = np.identity(n_eq + self.n_ineq)
    
        #  Check predicted violation reduction for search direction
        #  given by current penalty parameter
        d                   = self.solveSteeringDualQP()
        reduction           = predictedViolFn(d)
        if reduction >= c_viol*self.violation - violation_tol:
            return [d,mu,reduction]
        
    
        #  Disable steering if all inequality constraints are strictly 
        #  feasible, i.e., at least ineq_margin away from the feasible boundary,
        #  and no equality constraints are present.
        #  Explicitly check for infinity in case ineq contains -inf 
        if ineq_margin != np.inf and not np.any(self.ineq >= -ineq_margin) and n_eq == 0:
            return [d,mu,reduction]
        
            
        #  Predicted violation reduction was inadequate.  Check to see
        #  if reduction is an adequate fraction of the predicted reduction 
        #  when using the reference direction (given by the QP with the 
        #  objective removed, that is, with the penalty parameter temporarily 
        #  set to zero)
        self.updateSteeringQP(0)
        d_reference         = self.solveSteeringDualQP()
        reduction_reference = predictedViolFn(d_reference)
        if reduction >= c_viol*reduction_reference - violation_tol:
            return [d,mu,reduction]
        
    
        #  iteratively lower penalty parameter to produce new search directions
        #  which should hopefully have predicted reductions that are larger 
        #  fractions of the predicted reduction of the reference direction
        for j in range(maxit):
            mu = c_mu * mu
            self.updateSteeringQP(mu)
            d               = self.solveSteeringDualQP()
            reduction       = predictedViolFn(d)
            #  Test new step's predicted reduction against reference's
            if reduction >= c_viol*reduction_reference - violation_tol:
                return [d,mu,reduction] # predicted reduction is acceptable
            
        
        #  All d failed to meet acceptable predicted violation reduction so 
        #  just use the last search direction d produced for the lowest penalty 
        #  parameter value considered. 
        
        return [d,mu,reduction]
    
    #  private helper functions
    
    #  calculate predicted violation reduction for search direction d  using 
    #  a linearized constraint model
    
    #  l1 total violation
    def predictedViolationReductionL1(self,d):
        tmp_arr = self.ineq + self.ineq_grad.T @ d
        tmp_arr[tmp_arr < 0] = 0

        dL = self.violation - np.sum(tmp_arr) - LA.norm(self.eq + self.eq_grad.T@d,1)
        return dL

    #  l-infinity total violation
    def predictedViolationReduction(self,d):
        tmp_arr = self.ineq + self.ineq_grad.T @ d
        dL = self.violation - np.max(np.max(tmp_arr),0) - LA.norm(self.eq + self.eq_grad.T@d,ord = np.inf)
        return dL
    
    #  solve dual of steering QP to yeild new search direction
    #  throws an error if QP solver failed somehow
    def solveSteeringDualQP(self):
        try: 
            # qpalm only for linux
            if self.QPsolver == "gurobi":
                #  formulation of QP has no 1/2
                y = solveQP(self.H,self.f,None,None,self.LB,self.UB, "gurobi")
        except Exception as e:
            print(e)
            print("PyGRANSO steeringQuadprogFailure: Steering aborted due to a quadprog failure.")

        # dbg_print("Skip try & except in qpsteering strategy")
        

        d = -self.mu_Hinv_f_grad - (self.Hinv_c_grads @ y)
        return d
    
    #  update penalty parameter dependent values for QP
    def updateSteeringQP(self,mu):
        self.mu_Hinv_f_grad  = mu * self.Hinv_f_grad
        self.f               = conj(self.c_grads.T) @ self.mu_Hinv_f_grad - np.vstack((self.eq, self.ineq))
        return