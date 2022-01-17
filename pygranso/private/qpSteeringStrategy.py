import numpy as np
import numpy.linalg as LA
import torch
from pygranso.private.solveQP import solveQP
import traceback,sys

class qpSS:
    def __init__(self):
        pass

    def qpSteeringStrategy( self,penaltyfn_at_x, apply_Hinv, l1_model, ineq_margin, maxit, c_viol, c_mu, QPsolver, torch_device, double_precision):
        """
        qpSteeringStrategy:
            attempts to find a search direction which promotes progress towards
            feasibility.  

            INPUT:
               penaltyfn_at_x  struct containing fields for:
               .mu             current penalty parameter
               .f_grad         gradient of objective function at x
               .ci             inequality constraints evaluated at x 
               .ci_grad        corresponding gradients at x
               .ce             equality constraints evaluated at x
               .ce_grad        corresponding gradients at x
               .tv_l1          total violation value at x (one norm)
               .tv             total violation value at x (infinity norm)

               l1_model        logical: determines whether or not the one norm 
                               (the standard choice) or the infinity norm is used
                               for the total violation measure, which affects the
                               predicted violation reduction.

               ineq_margin     real value in [0,inf] setting the margin of 
                               feasibility for problems having only inequality 
                               constraints.  In this case, steering is selectively 
                               disabled when the inequality constraints are all at 
                               least ineq_margin away from being active.  Setting 
                               ineq_margin to zero means that steering will only 
                               be applied when one or more inequality constraints 
                               are active ( >= 0).  Setting ineq_margin to inf 
                               means that steering will be applied on every 
                               iteration.  NOTE: this parameter has no effect if 
                               equality constraints are present.

               apply_Hinv      function handle apply_Hinv(x)
                               returns b = Hinv*x where Hinv is the inverse
                               Hessian (or approximation to it).  Hinv must be
                               positive definite.  x may be a single column or
                               matrix.

               maxit           max iteration count to try lowering penalty 
                               parameter mu in order to find a good search 
                               direction

               c_viol          percentage of total violation needed to be acheived
                               by the predicted reduction for a candidate 
                               direction must be in (0,1)

               c_mu            scalar factor to reduce penalty paremeter mu on 
                               each iterative if resulting direction is not 
                               acceptable must be in (0,1)

               quadprog_opts   struct of options for quadprog interface.
                               It must be provided but it may be set as []

            OUTPUT:
               d               candidate search direction
                               d will be set to [] if all QP solves fail hard

               mu              possibly lower value of the penalty parameter 

               reduction       amount of total violation reduction d is predicted 
                               to yield via the linearized constraint model

            THROWS:
               error           if any call to quadprog either throws an error
                               (which will be set as .cause of the GRANSO error)
                               or if quadprog returns without error but its answer 
                               is numerically invalid (e.g. inf, nan, empty, zero) 

            If you publish work that uses or refers to PyGRANSO, please cite both
            PyGRANSO and GRANSO paper:

            [1] Buyun Liang, Tim Mitchell, and Ju Sun,
                NCVX: A User-Friendly and Scalable Package for Nonconvex
                Optimization in Machine Learning, arXiv preprint arXiv:2111.13984 (2021).
                Available at https://arxiv.org/abs/2111.13984

            [2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton,
                A BFGS-SQP method for nonsmooth, nonconvex, constrained
                optimization and its evaluation using relative minimization
                profiles, Optimization Methods and Software, 32(1):148-181, 2017.
                Available at https://dx.doi.org/10.1080/10556788.2016.1208749
                
            qpSteeringStrategy.py (introduced in PyGRANSO v1.0.0)
            Copyright (C) 2016-2021 Tim Mitchell and Buyun Liang

            This file is a MATLAB-to-Python port of qpSteeringStrategy.m from
            GRANSO v1.6.4 with the following new functionality and/or changes:
                1. Deriving necessary input arguments for OSQP and Gurobi interfaces 
            Ported from MATLAB to Python and modified by Buyun Liang, 2021

            For comments/bug reports, please visit the PyGRANSO webpage:
            https://github.com/sun-umn/PyGRANSO

            =========================================================================
            |  PyGRANSO: A PyTorch-enabled port of GRANSO with auto-differentiation |
            |  Copyright (C) 2021 Tim Mitchell and Buyun Liang                      |
            |                                                                       |
            |  This file is part of PyGRANSO.                                       |
            |                                                                       |
            |  PyGRANSO is free software: you can redistribute it and/or modify     |
            |  it under the terms of the GNU Affero General Public License as       |
            |  published by the Free Software Foundation, either version 3 of       |
            |  the License, or (at your option) any later version.                  |
            |                                                                       |
            |  PyGRANSO is distributed in the hope that it will be useful,          |
            |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
            |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
            |  GNU Affero General Public License for more details.                  |
            |                                                                       |
            |  You should have received a copy of the GNU Affero General Public     |
            |  License along with this program.  If not, see                        |
            |  <http://www.gnu.org/licenses/agpl.html>.                             |
            =========================================================================
        """
        self.device = torch_device
        self.double_precision = double_precision
        if double_precision:
            self.torch_dtype = torch.double
        else:
            self.torch_dtype = torch.float
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
        self.c_grads             = torch.hstack((self.eq_grad, self.ineq_grad))
        self.Hinv_c_grads        = apply_Hinv(self.c_grads)
        self.H                   = torch.conj(self.c_grads.t()) @ self.Hinv_c_grads
        #  Fix H since numerically, it is unlikely to be _perfectly_ symmetric 
        self.H                   = (self.H + torch.conj(self.H.t())) / 2
        self.mu_Hinv_f_grad      = mu * self.Hinv_f_grad
        self.f                   = torch.conj(self.c_grads.t()) @ self.mu_Hinv_f_grad - torch.vstack((self.eq, self.ineq)) 

        self.LB                  = torch.vstack((-torch.ones((n_eq,1)), torch.zeros((self.n_ineq,1))  )).to(device=self.device, dtype=self.torch_dtype)   
        self.UB                  = torch.ones((n_eq + self.n_ineq, 1),device=self.device, dtype=self.torch_dtype) 
        
        #  Identity matrix: compatible with the constraint form in QPALM
        self.A                   = torch.eye(n_eq + self.n_ineq,device=self.device, dtype=self.torch_dtype) 
    
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
        if ineq_margin != np.inf and not torch.any(self.ineq >= -ineq_margin) and n_eq == 0:
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

        dL = self.violation - torch.sum(tmp_arr) - torch.norm(self.eq + self.eq_grad.t()@d, p = 1)
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
                y = solveQP(self.H,self.f,None,None,self.LB,self.UB, "gurobi", self.device, self.double_precision)
            elif self.QPsolver == "osqp":
                #  formulation of QP has no 1/2
                y = solveQP(self.H,self.f,None,None,self.LB,self.UB, "osqp", self.device, self.double_precision)
        except Exception as e:
            print("PyGRANSO steeringQuadprogFailure: Steering aborted due to a quadprog failure.")        
            print(traceback.format_exc())
            sys.exit()

        d = -self.mu_Hinv_f_grad - (self.Hinv_c_grads @ y)
        return d
    
    #  update penalty parameter dependent values for QP
    def updateSteeringQP(self,mu):
        self.mu_Hinv_f_grad  = mu * self.Hinv_f_grad
        self.f               = torch.conj(self.c_grads.t()) @ self.mu_Hinv_f_grad - torch.vstack((self.eq, self.ineq))
        return