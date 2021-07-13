from private import pygransoConstants as pC, bfgsDamping as bD, regularizePosDefMatrix as rPDM, linesearchWeakWolfe as lWW
from private.neighborhoodCache import nC
from private.qpSteeringStrategy import qpSS
from private.qpTerminationCondition import qpTC
import time
from pygransoStruct import genral_struct
import math
import numpy.linalg as LA
import numpy as np

class AlgBFGSSQP():
    def __init__(self):
        pass

    def bfgssqp(self,penaltyfn_obj, bfgs_obj, opts, printer):
        """
        bfgssqp:
        Minimizes a penalty function.  Note that bfgssqp operates on the
        objects it takes as input arguments and bfgssqp will modify their
        states.  The result of bfgssqp's optimization process is obtained
        by querying these objects after bfgssqp has been run.
        """
        self.penaltyfn_obj = penaltyfn_obj
        self.bfgs_obj = bfgs_obj
        self.printer = printer

        #  "Constants" for controlling fallback levels
        #  currently these are 2 and 4 respectively
        [POSTQP_FALLBACK_LEVEL, self.LAST_FALLBACK_LEVEL] = pC.pygransoConstants()
                    
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

        if full_memory and self.regularize_threshold < float("inf"):
            get_apply_H_QP_fn   = lambda : self.getApplyHRegularized()
        else:
            #  No regularization option for limited memory BFGS 
            get_apply_H_QP_fn   = lambda : self.getApplyH()

        [self.apply_H_QP_fn, H_QP]   = get_apply_H_QP_fn()
        #  For applying the normal non-regularized version of H
        [self.apply_H_fn,*_]            = self.getApplyH()  
        
        self.bfgs_update_fn          = lambda : self.bfgs_obj.update()

        #  function which caches up to ngrad previous gradients and will return 
        #  those which are sufficently close to the current iterate x. 
        #  The gradients from the current iterate are simultaneously added to 
        #  the cache. 
        nC_obj = nC()
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
                                steering_c_mu,      self.QPsolver           )

        self.linesearch_fn   = lambda x,f,g,p,ls_maxit: lWW.linesearchWeakWolfe( 
                                x, f, g, p,                                  
                                lambda x_in: self.penaltyfn_obj.evaluatePenaltyFunction(x_in),      
                                wolfe1, wolfe2, self.fvalquit, ls_maxit, step_tol)
                                                        
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
                
                # try:
                #     [p,mu_new] = steering_fn(self.penaltyfn_at_x,apply_H_steer)
                # except Exception as e:
                #     print(e)
                #     print("PyGRANSO:steeringQuadprogFailure")
                print("Skip try & except in bfgssqp")
                [p,mu_new,*_] = steering_fn(self.penaltyfn_at_x,apply_H_steer)

                penalty_parameter_changed = (mu_new != self.mu)
                if penalty_parameter_changed: 
                    [f,g,self.mu] = self.penaltyfn_obj.updatePenaltyParameter(mu_new)
                
            elif self.fallback_level == 2:
                p = -self.apply_H_fn(g)   # standard BFGS 
            elif self.fallback_level == 3:
                p = -g;     # steepest descent
            else:
                p = np.random.randn(n,1)
                self.random_attempts = self.random_attempts + 1
                
                
            [p,is_descent,fallback_on_this_direction] = self.checkDirection(p,g)

            if fallback_on_this_direction:
                if self.bumpFallbackLevel():
                    continue    # try iteration again with new fallback
                else: # all fallbacks have failed - quit!
                    self.prepareTermination(9); # not a descent descent direction
                    return self.info
                end
            else: # ATTEMPT LINE SEARCH
                f_prev = f      # for relative termination tolerance
                self.g_prev = g      # necessary for BFGS update
                if is_descent:
                    ls_procedure_fn = lambda x,f,g,p: self.linesearchDescent(x,f,g,p)
                else:
                    ls_procedure_fn = lambda x,f,g,p: self.linesearchNondescent(x,f,g,p)
                
                # this will also update gprev if it lowers mu and it succeeds
                [alpha,x_new,f,g,linesearch_failed] = ls_procedure_fn(x,f,g,p)
            
                
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
                        self.prepareTermination(6 + ~feasible);
                        return self.info
                    
                elif linesearch_failed == 2:  # couldn't bracket minimizer
                    self.prepareTermination(8);
                    return self.info
                else: # failed on nondescent direction
                    if can_fallback:
                        continue
                    else:
                        self.prepareTermination(9);
                    
                    
            
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
            
        
            if self.print_level and (iter % print_frequency) == 0:
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
        gtp                 = g.T@p
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
        [alpha,x,f,g,fail] = self.linesearch_fn( x,f,g,p,self.linesearch_nondescent_maxit )
        fail = 0 + 3*(fail > 0)
        return [alpha, x, f, g, fail,_,_,_]

    #  regular weak Wolfe line search 
    #  NOTE: this function may lower variable "mu" for constrained problems
    def linesearchDescent(self,x,f,g,p):
        
        #  we need to keep around f and g so use _ls names for ls results
        ls_fn                       = lambda f,g: self.linesearch_fn(x,f,g,p,float("inf"))
        [alpha,x_ls,f_ls,g_ls,fail,_,_,_] = ls_fn(f,g)
                        
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
        stat_value      = LA.norm(stat_vec)
        if stat_value <= self.opt_tol:
            n_qps       = 0
            n_samples   = 1
            dist_evals  = 0
            self.penaltyfn_obj.addStationarityMeasure(stat_value)
            return
      
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
                                                        self.apply_H_QP_fn, self.QPsolver)
        stat_value = LA.norm(stat_vec)
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
        self.info.termination_code   = code
        if code == 8 and self.constrained:
            self.info.mu_lowest      = self.mu_lowest

    def applyBfgsUpdate(self,alpha,p,g,gprev):
                    
        s               = alpha*p
        y               = g - gprev
        sty             = s.T@y
        
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
    grads = genral_struct()
    setattr(grads,"F",f_grad)
    setattr(grads,"CI",ci_grad)
    setattr(grads,"CE",ce_grad)
    [*_,grads,dist_evals] = grad_nbd_fn(penaltyfn_obj.getX(), grads)
    return [grads,dist_evals]