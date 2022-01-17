import numpy as np
import torch
from pygranso.private.solveQP import solveQP
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

            INPUT:
               penaltyfn_at_x      struct containing fields for:
               .mu                 current penalty parameter
               .f                  objective function evaluated at x
               .ci                 inequality constraints evaluated at x 
               .ce                 equality constraints evaluated at x

               gradient_samples    a cell array of structs containing fields
                                   'F', 'CI', and 'CE', which contain the
                                   respective gradients for a history or 
                                   collection of the objective
                                   function, the inequality constraints, and
                                   equality constraints, evaluated at different 
                                   x_k.  Each index of the cell array contains 
                                   these F, CI, and CE values for a different
                                   value of x_k.  One of these x_k should
                                   correspond to x represented in penaltyfn_parts

               apply_Hinv          function handle apply_Hinv(x)
                                   returns b = Hinv*x where Hinv is the inverse
                                   Hessian (or approximation to it).  Hinv must be
                                   positive definite.  x may be a single column or
                                   matrix.

               quadprog_opts       struct of options for quadprog interface
                                   It must be provided but it may be set as []

            OUTPUT:
               d                   smallest vector in convex hull of gradient 
                                   samples d is a vector of Infs if QP solver 
                                   fails hard

               qps_solved          number of QPs attempted in order to compute
                                   some variant of d.  If quadprog fails when
                                   attempting to compute d, up to three
                                   alternative QP formulations will be attempted.

               ME                  empty [] if default d was computed normally.
                                   Otherwise an MException object containing
                                   the causes of why each method of computing d 
                                   failed, accrued in the .cause field.        

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
                
            qpTerminationCondition.py (introduced in PyGRANSO v1.0.0)
            Copyright (C) 2016-2021 Tim Mitchell

            This is a direct port of qpTerminationCondition.m from GRANSO v1.6.4.
            Ported from MATLAB to Python by Buyun Liang 2021.

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
            print("PyGRANSO:qpTerminationCondition type 1 failure")
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
            print("PyGRANSO:qpTerminationCondition type 2 failure")
            print(traceback.format_exc())
    
        # % Fall back strategy #2: revert to MATLAB's quadprog, if user is
        # % using a different quadprog solver and reattempt with original H
        # Skip in PyGRANSO

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
            print("PyGRANSO:qpTerminationCondition type 4 failure")
            print(traceback.format_exc())
