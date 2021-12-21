import torch
from private import ncvxConstants as pC, bfgsDamping as bD, regularizePosDefMatrix as rPDM, linesearchWeakWolfe as lWW
from private.neighborhoodCache import nC
from private.qpSteeringStrategy import qpSS
from private.qpTerminationCondition import qpTC
import time
from ncvxStruct import GeneralStruct
import math
import numpy as np
from numpy.random import default_rng
import traceback,sys

class AlgBFGSSQP():
    def __init__(self):
        pass

    def bfgssqp(self,f_eval_fn, penaltyfn_obj, bfgs_obj, opts, printer, torch_device):
        """
        bfgssqp:
            Minimizes a penalty function.  Note that bfgssqp operates on the
            objects it takes as input arguments and bfgssqp will modify their
            states.  The result of bfgssqp's optimization process is obtained
            by querying these objects after bfgssqp has been run.

            INPUT:
                penaltyfn_obj       
                    Penalty function object from makePenaltyFunction.m

                bfgs_obj
                    (L)BFGS object from bfgsHessianInverse.m or
                    bfgsHessianInvereLimitedMem.m

                opts
                    A struct of parameters for the software.  See gransoOptions.m

                printer
                    A printer object from gransoPrinter.m

            OUTPUT:
                info    numeric code indicating termination circumstance:

                    0:  Approximate stationarity measurement <= opts.opt_tol and 
                        current iterate is sufficiently close to the feasible 
                        region (as determined by opts.viol_ineq_tol and 
                        opts.viol_eq_tol).

                    1:  Relative decrease in penalty function <= opts.rel_tol and
                        current iterate is sufficiently close to the feasible 
                        region (as determined by opts.viol_ineq_tol and 
                        opts.viol_eq_tol).

                    2:  Objective target value reached at an iterate
                        sufficiently close to feasible region (determined by
                        opts.fvalquit, opts.viol_ineq_tol and opts.viol_eq_tol).

                    3:  User requested termination via opts.halt_log_fn 
                        returning true at this iterate.

                    4:  Max number of iterations reached (opts.maxit).

                    5:  Clock/wall time limit exceeded (opts.maxclocktime).

                    6:  Line search bracketed a minimizer but failed to satisfy
                        Wolfe conditions at a feasible point (with respect to
                        opts.viol_ineq_tol and opts.viol_eq_tol).  For 
                        unconstrained problems, this is often an indication that a
                        stationary point has been reached.

                    7:  Line search bracketed a minimizer but failed to satisfy
                        Wolfe conditions at an infeasible point.

                    8:  Line search failed to bracket a minimizer indicating the
                        objective function may be unbounded below.  For constrained
                        problems, where the objective may only be unbounded off the
                        feasible set, consider restarting NCVX with opts.mu0 set
                        lower than soln.mu_lowest (see its description below for 
                        more details).

                    9:  NCVX failed to produce a descent direction.

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
                bfgssqp.m introduced in GRANSO Version 1.0
                
                Buyun Dec 20, 2021 (NCVX Version 1.0.0):
                    bfgssqp.py is translated from bfgssqp.m in GRANSO Version 1.6.4. 

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
        self.f_eval_fn = f_eval_fn
        self.penaltyfn_obj = penaltyfn_obj
        self.bfgs_obj = bfgs_obj
        self.printer = printer

        #  "Constants" for controlling fallback levels
        #  currently these are 2 and 4 respectively
        [POSTQP_FALLBACK_LEVEL, self.LAST_FALLBACK_LEVEL] = pC.ncvxConstants()
                    
        #  initialization parameters
        x                           = opts.x0
        n                           = len(x)
        full_memory                 = opts.limited_mem_size == 0
        self.damping                     = opts.bfgs_damping
        
        #  convergence criteria termination parameters
        #  violation tolerances are checked and handled by penaltyfn_obj
        self.opt_tol                     = opts.opt_tol
        self.rel_tol                     = opts.rel_tol
        step_tol                    = opts.step_tol
        ngrad                       = opts.ngrad
        evaldist                    = opts.evaldist

        #  early termination parameters
        maxit                       = opts.maxit
        maxclocktime                = opts.maxclocktime
        if maxclocktime < float("inf"):       
            t_start = time.time()
        
        self.fvalquit                    = opts.fvalquit
        halt_on_quadprog_error      = opts.halt_on_quadprog_error
        halt_on_linesearch_bracket  = opts.halt_on_linesearch_bracket

        #  fallback parameters - allowable last resort "heuristics"
        min_fallback_level          = opts.min_fallback_level
        self.max_fallback_level          = opts.max_fallback_level
        self.max_random_attempts         = opts.max_random_attempts

        #  steering parameters
        steering_l1_model           = opts.steering_l1_model
        steering_ineq_margin        = opts.steering_ineq_margin
        steering_maxit              = opts.steering_maxit
        steering_c_viol             = opts.steering_c_viol
        steering_c_mu               = opts.steering_c_mu
        
        #  parameters for optionally regularizing of H 
        self.regularize_threshold        = opts.regularize_threshold
        self.regularize_max_eigenvalues  = opts.regularize_max_eigenvalues
        
        self.QPsolver               = opts.QPsolver

        #  line search parameters
        wolfe1                      = opts.wolfe1
        wolfe2                      = opts.wolfe2
        self.linesearch_nondescent_maxit = opts.linesearch_nondescent_maxit
        self.linesearch_reattempts       = opts.linesearch_reattempts
        self.linesearch_reattempts_x0    = opts.linesearch_reattempts_x0
        self.linesearch_c_mu             = opts.linesearch_c_mu
        self.linesearch_c_mu_x0          = opts.linesearch_c_mu_x0

        self.linesearch_maxit = opts.linesearch_maxit
        self.init_step_size = opts.init_step_size
        self.is_backtrack_linesearch = opts.is_backtrack_linesearch
        self.searching_direction_rescaling = opts.searching_direction_rescaling
        self.disable_terminationcode_6 = opts.disable_terminationcode_6
        self.double_precision = opts.double_precision
        if self.double_precision:
            self.torch_dtype = torch.double
        else:
            self.torch_dtype = torch.float

        #  logging parameters
        self.print_level                 = opts.print_level
        print_frequency             = opts.print_frequency
        halt_log_fn                 = opts.halt_log_fn
        user_halt                   = False

        #  get value of penalty function at initial point x0 
        #  mu will be fixed to one if there are no constraints.
        self.iter            = 0
        [f,g]           = self.penaltyfn_obj.getPenaltyFunctionValue()
        self.mu              = self.penaltyfn_obj.getPenaltyParameter()
        self.constrained     = self.penaltyfn_obj.hasConstraints()

        #  The following will save all the function values and gradients for
        #  the objective, constraints, violations, and penalty function
        #  evaluated at the current x and value of the penalty parameter mu.
        #  If a search direction fails, the code will "roll back" to this data
        #  (rather than recomputing it) in order to attempt one of the fallback
        #  procedures for hopefully obtaining a usable search direction.
        #  NOTE: bumpFallbackLevel() will restore the last executed snapshot. It
        #  does NOT need to be provided.
        self.penaltyfn_at_x  = self.penaltyfn_obj.snapShot()
                                            
        #  regularizes H for QP solvers but only if cond(H) > regularize_limit
        #  if isinf(regularize_limit), no work is done

        self.torch_device = torch_device

        if full_memory and self.regularize_threshold < float("inf"):
            get_apply_H_QP_fn   = lambda : self.getApplyHRegularized()
        else:
            #  No regularization option for limited memory BFGS 
            get_apply_H_QP_fn   = lambda : self.getApplyH()

        [self.apply_H_QP_fn, H_QP]   = get_apply_H_QP_fn()
        #  For applying the normal non-regularized version of H
        [self.apply_H_fn,*_]            = self.getApplyH()  
        
        self.bfgs_update_fn          = lambda s,y,sty,damped: self.bfgs_obj.update(s,y,sty,damped)

        #  function which caches up to ngrad previous gradients and will return 
        #  those which are sufficently close to the current iterate x. 
        #  The gradients from the current iterate are simultaneously added to 
        #  the cache. 
        nC_obj = nC(torch_device,self.double_precision)
        get_nbd_grads_fn        = nC_obj.neighborhoodCache(ngrad,evaldist)
        self.get_nearby_grads_fn     = lambda : getNearbyGradients( self.penaltyfn_obj, get_nbd_grads_fn)

        [stat_vec,self.stat_val,qps_solved, _, _] = self.computeApproxStationarityVector()
    
        if not self.constrained:
            #  disable steering QP solves by increasing min_fallback_level.
            min_fallback_level  = max(  min_fallback_level, POSTQP_FALLBACK_LEVEL )
            #  make sure max_fallback_level is at min_fallback_level
            self.max_fallback_level  = max(min_fallback_level, self.max_fallback_level)
        
        if self.max_fallback_level > 0: 
            APPLY_IDENTITY      = lambda x: x
        
        if np.any(halt_log_fn!=None):
            get_bfgs_state_fn = lambda : self.bfgs_obj.getState()
            user_halt = halt_log_fn(0, x, self.penaltyfn_at_x, np.zeros((n,1)),
                                    get_bfgs_state_fn, H_QP,
                                    1, 0, 1, stat_vec, self.stat_val, 0          )
        

        if self.print_level:
            self.printer.init(self.penaltyfn_at_x,self.stat_val,qps_solved)
        

        self.rel_diff = float("inf")
        if self.converged():
            return self.info
        elif user_halt:
            self.prepareTermination(3)
            return self.info

        #  set up a more convenient function handles to reduce fixed arguments
        qpSS_obj = qpSS()
        steering_fn     = lambda penaltyfn_parts,H: qpSS_obj.qpSteeringStrategy(
                                penaltyfn_parts,    H, 
                                steering_l1_model,  steering_ineq_margin, 
                                steering_maxit,     steering_c_viol, 
                                steering_c_mu,      self.QPsolver, torch_device, self.double_precision           )

        self.linesearch_fn   = lambda x,f,g,p,ls_maxit: lWW.linesearchWeakWolfe( 
                                x, f, g, p,
                                lambda x_in: self.penaltyfn_obj.evaluatePenaltyFunction4linesearch(x_in),                                  
                                lambda x_in: self.penaltyfn_obj.evaluatePenaltyFunction(x_in),      
                                wolfe1, wolfe2, self.fvalquit, ls_maxit, step_tol, self.init_step_size, self.linesearch_maxit, self.is_backtrack_linesearch, self.torch_device)
                                                        
        #  we'll use a while loop so we can explicitly update the counter only
        #  for successful updates.  This way, if the search direction direction
        #  can't be used and the code falls back to alternative method to try
        #  a new search direction, the iteration count is not increased for
        #  these subsequent fallback attempts
        
        #  loop control variables
        self.fallback_level      = min_fallback_level
        self.random_attempts     = 0
        self.iter                = 1
        evals_so_far        = self.penaltyfn_obj.getNumberOfEvaluations()

        while self.iter <= maxit:
            # Call standard steering strategy to produce search direction p
            # which hopefully "promotes progress towards feasibility".
            # However, if the returned p is empty, this means all QPs failed
            # hard.  As a fallback, steering will be retried with steepest
            # descent, i.e. H temporarily  set to the identity.  If this
            # fallback also fails hard, then the standard BFGS search direction
            # on penalty function is tried.  If this also fails, then steepest
            # will be tried.  Finally, if all else fails, randomly generated
            # directions are tried as a last ditch effort.
            # NOTE: min_fallback_level and max_fallback_level control how much
            #     of this fallback range is available.

            # NOTE: the penalty parameter is only lowered by two actions:
            # 1) either of the two steering strategies lower mu and produce
            # a step accepted by the line search
            # 2) a descent direction (generated via any fallback level) is not
            # initially accepted by the line search but a subsequent
            # line search attempt with a lowered penalty parameter does
            # produce an accepted step.

            penalty_parameter_changed = False
            if self.fallback_level < POSTQP_FALLBACK_LEVEL:  
                if self.fallback_level == 0:
                    apply_H_steer = self.apply_H_QP_fn;  # standard steering   
                else:
                    apply_H_steer = APPLY_IDENTITY; # "degraded" steering 
                
                try:
                    [p,mu_new,*_] = steering_fn(self.penaltyfn_at_x,apply_H_steer)
                except Exception as e:
                    print("NCVX:steeringQuadprogFailure")
                    print(traceback.format_exc())
                    sys.exit()
                
                penalty_parameter_changed = (mu_new != self.mu)
                if penalty_parameter_changed: 
                    [f,g,self.mu] = self.penaltyfn_obj.updatePenaltyParameter(mu_new)
                
            elif self.fallback_level == 2:
                p = -self.apply_H_fn(g)   # standard BFGS 
            elif self.fallback_level == 3:
                p = -g;     # steepest descent

                
            else:
                rng = default_rng()
                p = rng.standard_normal(size=(n,1))
                self.random_attempts = self.random_attempts + 1
            
            if self.searching_direction_rescaling:
                p_norm = torch.norm(p).item()
                p =  1 * p / p_norm
                
            [p,is_descent,fallback_on_this_direction] = self.checkDirection(p,g)

            if fallback_on_this_direction:
                if self.bumpFallbackLevel():
                    continue    # try iteration again with new fallback
                else: # all fallbacks have failed - quit!
                    self.prepareTermination(9); # not a descent descent direction
                    return self.info
                
            else: # ATTEMPT LINE SEARCH
                f_prev = f      # for relative termination tolerance
                self.g_prev = g      # necessary for BFGS update
                if is_descent:
                    ls_procedure_fn = lambda x,f,g,p: self.linesearchDescent(x,f,g,p)
                else:
                    ls_procedure_fn = lambda x,f,g,p: self.linesearchNondescent(x,f,g,p)
                
                # this will also update gprev if it lowers mu and it succeeds
                [alpha,x_new,f,g,linesearch_failed] = ls_procedure_fn(x,f,g,p)

            if self.disable_terminationcode_6:
                if linesearch_failed and self.fallback_level == 3:
                    linesearch_failed = 0

            if linesearch_failed:
                #  first get lowest mu attempted (restore will erase it)
                self.mu_lowest = self.penaltyfn_obj.getPenaltyParameter()
                #  now, for all failure types, restore last accepted iterate
                can_fallback = self.bumpFallbackLevel()
                if linesearch_failed == 1:  # bracketed minimizer but LS failed
                    feasible = self.penaltyfn_obj.isFeasibleToTol()
                    if halt_on_linesearch_bracket and feasible:
                        self.prepareTermination(6)
                        return self.info
                    elif can_fallback:
                        continue
                    else: # return 6 (feasible) or 7 (infeasible)
                        self.prepareTermination(7 - feasible)
                        return self.info
                    
                elif linesearch_failed == 2:  # couldn't bracket minimizer
                    self.prepareTermination(8)
                    return self.info
                else: # failed on nondescent direction
                    if can_fallback:
                        continue
                    else:
                        self.prepareTermination(9)
                    
                    
            
            # ELSE LINE SEARCH SUCCEEDED - STEP ACCEPTED
            
            # compute relative difference of change in penalty function values
            # this will be infinity if previous value was 0 or if the value of
            # the penalty parameter was changed
            if penalty_parameter_changed or f_prev == 0:
                self.rel_diff = float("inf")
            else:
                self.rel_diff = abs(f - f_prev) / abs(f_prev)
     
            # update x to accepted iterate from line search
            # mu is already updated by line search if lowered
            x = x_new

            # Update all components of the penalty function evaluated
            # at the new accepted iterate x and snapsnot the data.
            self.penaltyfn_at_x          = self.penaltyfn_obj.snapShot()

            # for stationarity condition
            [ stat_vec, self.stat_val, qps_solved, n_grad_samples,_]   = self.computeApproxStationarityVector()
                
            
            ls_evals = self.penaltyfn_obj.getNumberOfEvaluations()-evals_so_far
            
            # Perform full or limited memory BFGS update
            # This computation is done before checking the termination
            # conditions because we wish to provide the most recent (L)BFGS
            # data to users in case they desire to restart.   
            self.applyBfgsUpdate(alpha,p,g,self.g_prev)
        
            if np.any(halt_log_fn!=None):
                user_halt = halt_log_fn(self.iter, x, self.penaltyfn_at_x, p, 
                                        get_bfgs_state_fn, H_QP, 
                                        ls_evals, alpha, n_grad_samples, 
                                        stat_vec, self.stat_val, self.fallback_level  )
            
        
            if self.print_level and (self.iter % print_frequency) == 0:
                self.printer.iter(   self.iter, self.penaltyfn_at_x, 
                                self.fallback_level, self.random_attempts,  
                                ls_evals,       alpha,  
                                n_grad_samples, self.stat_val,   qps_solved  );     
  
                
            # reset fallback level counters
            self.fallback_level  = min_fallback_level
            self.random_attempts = 0
            evals_so_far    = self.penaltyfn_obj.getNumberOfEvaluations()
    
            #  check convergence/termination conditions
            if self.converged():
                return self.info
            elif user_halt:
                self.prepareTermination(3)
                return self.info
            elif maxclocktime < float("inf") and (time.time()-t_start) > maxclocktime:
                self.prepareTermination(5)
                return self.info
            
            
            #  if cond(H) > regularize_limit, make a regularized version of H
            #  for QP solvers to use on next iteration
            if self.iter < maxit:     # don't bother if maxit has been reached
                [self.apply_H_QP_fn, H_QP] = get_apply_H_QP_fn()
            

            self.iter = self.iter + 1   # only increment counter for successful updates
        # end while loop

        self.prepareTermination(4)  # max iterations reached

        return self.info

    #  PRIVATE NESTED FUNCTIONS
    
    def checkDirection(self,p,g):    
        fallback            = False
        gtp                 = torch.conj(g.t())@p
        gtp = gtp.item()
        if math.isnan(gtp) or math.isinf(gtp):
            is_descent      = False
            fallback        = True    
        else:
            if gtp > 0 and self.fallback_level == self.LAST_FALLBACK_LEVEL:
                #  randomly generated ascent direction, flip sign of p
                p           = -p
                is_descent  = True
            else:
                is_descent  = gtp < 0
            
            if not is_descent and self.linesearch_nondescent_maxit == 0:
                fallback    = True
            
        return [p,is_descent,fallback]

    def bumpFallbackLevel(self):
        self.penaltyfn_obj.restoreSnapShot()
       
        can_fallback        = self.fallback_level < self.max_fallback_level
        if can_fallback:
            self.fallback_level  = self.fallback_level + 1
        elif self.fallback_level == self.LAST_FALLBACK_LEVEL and self.random_attempts < self.max_random_attempts:
            can_fallback    = True
        
        return can_fallback

    #  only try a few line search iterations if p is not a descent direction
    def linesearchNondescent(self,x,f,g,p):
        # [alpha,x,f,g,fail,_,_,_] = self.linesearch_fn( x,f,g,p,self.linesearch_nondescent_maxit )
        [alpha,x,f,g,fail] = self.linesearch_fn( x,f,g,p,self.linesearch_nondescent_maxit )
        fail = 0 + 3*(fail > 0)
        return [alpha, x, f, g, fail]

    #  regular weak Wolfe line search 
    #  NOTE: this function may lower variable "mu" for constrained problems
    def linesearchDescent(self,x,f,g,p):
        
        #  we need to keep around f and g so use _ls names for ls results
        ls_fn                       = lambda f,g: self.linesearch_fn(x,f,g,p,float("inf"))
        # [alpha,x_ls,f_ls,g_ls,fail,_,_,_] = ls_fn(f,g)
        [alpha,x_ls,f_ls,g_ls,fail] = ls_fn(f,g)
                        
        #  If the problem is constrained and the line search fails without 
        #  bracketing a minimizer, it may be because the objective is 
        #  unbounded below off the feasible set.  In this case, we can retry
        #  the line search with progressively lower values of mu.   
        if self.constrained and fail == 2:
        
            mu_ls = self.mu  # the original value of the penalty parameter
           
            if self.iter < 2: 
                reattempts  = self.linesearch_reattempts_x0
                ls_c_mu     = self.linesearch_c_mu_x0
            else:
                reattempts  = self.linesearch_reattempts
                ls_c_mu     = self.linesearch_c_mu
              
            for j in range(reattempts):
                #  revert to last good iterate (since line search failed)
                self.penaltyfn_obj.restoreSnapShot()
                #  lower the trial line search penalty parameter further
                mu_ls       = ls_c_mu * mu_ls
                [f,g]       = self.penaltyfn_obj.updatePenaltyParameter(mu_ls)
                gprev_ls    = g
                
                if self.print_level > 1:
                    self.printer.lineSearchRestart(self.iter,mu_ls)
                
                [alpha,x_ls,f_ls,g_ls,failed_again] = ls_fn(f,g)
               
                if not failed_again: # LINE SEARCH SUCCEEDED 
                    #  make sure mu and and gprev are up-to-date, since the
                    #  penalty parameter has been lowered 
                    fail    = False
                    self.mu      = self.penaltyfn_obj.getPenaltyParameter()
                    self.g_prev  = gprev_ls
                    return
       
        #  LINE SEARCH EITHER SUCCEEDED OR FAILED
        #  no need to restore snapshot if line search failed since 
        #  bumpFallbackLevel() will be called and it requests the last  
        #  snapshot to be restored.
        return [alpha, x_ls, f_ls, g_ls, fail]

    def computeApproxStationarityVector(self):
            
        #  first check the smooth case (gradient of the penalty function).
        #  If its norm is small, that indicates that we are at a smooth 
        #  stationary point and we can return this measure and terminate
        stat_vec        = self.penaltyfn_at_x.p_grad
        stat_value      = torch.norm(stat_vec)

        
        self.opt_tol = torch.as_tensor(self.opt_tol,device = self.torch_device, dtype=self.torch_dtype)

        if stat_value <= self.opt_tol:
            n_qps       = 0
            n_samples   = 1
            dist_evals  = 0
            self.penaltyfn_obj.addStationarityMeasure(stat_value)
            return [  stat_vec, stat_value, n_qps, n_samples, dist_evals ]
      
        #  otherwise, we must do a nonsmooth stationary point test
            
        #  add new gradients at current iterate to cache and then get
        #  all nearby gradients samples from history that are
        #  sufficiently close to current iterate (within a ball of
        #  radius evaldist centered at the current iterate x)
        [grad_samples,dist_evals] = self.get_nearby_grads_fn()
        
        #  number of previous iterates that are considered sufficiently
        #  close, including the current iterate
        n_samples = len(grad_samples)
        
        #  nonsmooth optimality measure
        qPTC_obj = qpTC()
        [stat_vec,n_qps,ME] = qPTC_obj.qpTerminationCondition(   self.penaltyfn_at_x, grad_samples,
                                                        self.apply_H_QP_fn, self.QPsolver, self.torch_device, self.double_precision)
        stat_value = torch.norm(stat_vec).item()
        self.penaltyfn_obj.addStationarityMeasure(stat_value)
        
        if self.print_level > 2 and  len(ME) > 0:
            self.printer.qpError(self.iter,ME,'TERMINATION')
        
        return [  stat_vec, stat_value, n_qps, n_samples, dist_evals ]

    def converged(self):
        tf = True
        #  only converged if point is feasible to tolerance
        if self.penaltyfn_at_x.feasible_to_tol:
            if self.stat_val <= self.opt_tol:
                self.prepareTermination(0)
                return tf
            elif self.rel_diff <= self.rel_tol:
                self.prepareTermination(1)   
                return tf
            elif self.penaltyfn_at_x.f <= self.fvalquit:
                self.prepareTermination(2)
                return tf
        tf = False
        return tf

    def prepareTermination(self,code):
        self.info = GeneralStruct()
        setattr(self.info, "termination_code", code)
        if code == 8 and self.constrained:
            self.info.mu_lowest      = self.mu_lowest

    def applyBfgsUpdate(self,alpha,p,g,gprev):
                    
        s               = alpha*p
        y               = g - gprev
        sty             = torch.conj(s.t())@y
        
        if self.damping > 0:
            [y,sty,damped] = bD.bfgsDamping(self.damping,self.apply_H_fn,s,y,sty)
        
        update_code     = self.bfgs_update_fn(s,y,sty,damped)
        
        if update_code > 0 and self.print_level > 1:
            self.printer.bfgsInfo(self.iter,update_code)
        

    def getApplyH(self):
        applyH  = self.bfgs_obj.applyH
        H       = None
        return [applyH, H]

    def getApplyHRegularized(self):
        #  This should only be called when running full memory BFGS as
        #  getState() only returns the inverse Hessian as a dense matrix in
        #  this case.  For L-BFGS, getState() returns a struct of data.
        [Hr,code] = rPDM.regularizePosDefMatrix( self.bfgs_obj.getState(),self.regularize_threshold,  
                                            self.regularize_max_eigenvalues  )
        if code == 2 and self.print_level > 2:
            self.printer.regularizeError(iter)
            
        applyHr  = lambda x: Hr@x
        
        #  We only return Hr so that it may be passed to the halt_log_fn,
        #  since (advanced) users may wish to look at it.  However, if
        #  regularization was actually not applied, i.e. H = Hr, then we can
        #  set Hr = [].  Users can already get H since @bfgs_obj.getState
        #  is passed into halt_log_fn and the [] value will indicate to the 
        #  user that regularization was not applied (which can be checked
        #  more efficiently and quickly than comparing two matrices).   
        if code == 1:
            Hr = None   
        
        return [applyHr, Hr]


def getNearbyGradients(penaltyfn_obj,grad_nbd_fn):
    [f_grad, ci_grad, ce_grad] = penaltyfn_obj.getGradients()
    grads = GeneralStruct()
    setattr(grads,"F",f_grad)
    setattr(grads,"CI",ci_grad)
    setattr(grads,"CE",ce_grad)
    [*_,grads_ret,dist_evals] = grad_nbd_fn(penaltyfn_obj.getX(), grads)
    return [grads_ret,dist_evals]