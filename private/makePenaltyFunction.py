import types
import numpy as np
import torch
from ncvxStruct import GeneralStruct
import traceback,sys

class PanaltyFuctions:
    def __init__(self):
        pass

    def makePenaltyFunction(self,params,f_eval_fn,obj_fn,varargin=None,torch_device = torch.device('cpu')):
        """
        makePenaltyFunction: 
            creates an object representing the penalty function for 
                min obj_fn 
                subject to ineq_fn <= 0
                                eq_fn == 0
            where the penalty function is specified with initial penalty 
            parameter value mu and is applied to the objective function.
            Roughly, this means:
                mu*obj_fn   + sum of active inequality constraints 
                            + sum of absolute value of eq. constraints
        """
        self.f_eval_fn = f_eval_fn 
        self.obj_fn = obj_fn 
        
        # local storage for function and gradients and the current x
        
        self.x = params.x0
        n = len(self.x)

        # objective and its gradient
    
        # originally: if nargin < 4. Currently we only allow combinefns
        try: 
            splitEvalAtX_obj = Class_splitEvalAtX()
            [self.f,self.f_grad,self.obj_fn,ineq_fn,eq_fn] = splitEvalAtX_obj.splitEvalAtX(self.obj_fn,self.x)
        except Exception as e:
            print(traceback.format_exc())
            sys.exit()

        assertFnOutputs(n,self.f,self.f_grad,'objective')

        prescaling_threshold = params.prescaling_threshold
        # checking scaling of objective and rescale if necessary
        f_grad_norm = torch.norm(self.f_grad)
        if f_grad_norm > prescaling_threshold:
            self.scaling_f = prescaling_threshold / f_grad_norm
            self.obj_fn = lambda x : rescaleObjective(x,self.obj_fn,self.scaling_f)
            self.f = self.f * self.scaling_f
            self.f_grad = self.f_grad * self.scaling_f
        else:
            self.scaling_f = None

        # setup inequality and equality constraints, violations, and scalings
        
        [ self.eval_ineq_fn,self.ci,self.ci_grad,self.tvi,self.tvi_l1,self.tvi_l1_grad,ci_grad_norms,self.scaling_ci,ineq_constrained] = setupConstraint(self.x,ineq_fn,lambda x, fn: self.evalInequality(x,fn),True,prescaling_threshold,torch_device)

        [self.eval_eq_fn,self.ce,self.ce_grad,self.tve,self.tve_l1,self.tve_l1_grad,ce_grad_norms,self.scaling_ce,eq_constrained] = setupConstraint(self.x,eq_fn,lambda x, fn: self.evalEquality(x,fn),False,prescaling_threshold,torch_device)

        grad_norms_at_x0 = GeneralStruct()
        setattr(grad_norms_at_x0,"f",f_grad_norm)
        setattr(grad_norms_at_x0,"ci",ci_grad_norms)
        setattr(grad_norms_at_x0,"ce",ce_grad_norms)

        self.scalings        = GeneralStruct()
        if np.any(self.scaling_f != None):
            setattr(self.scalings,"f",self.scaling_f)
        if np.any(self.scaling_ci != None):
            setattr(self.scalings,"ci",self.scaling_ci)
        if np.any(self.scaling_ce != None):
            setattr(self.scalings,"ce",self.scaling_ce)
        self.prescaled       = len(self.scalings.__dict__) != 0 

        constrained = ineq_constrained or eq_constrained
        if constrained:
            self.mu                          = params.mu0
            update_penalty_parameter_fn = lambda mu_new: self.updatePenaltyParameter(mu_new)
            self.viol_ineq_tol               = params.viol_ineq_tol
            self.viol_eq_tol                 = params.viol_eq_tol
            self.is_feasible_to_tol_fn       = lambda tvi, tve: self.isFeasibleToTol(tvi,tve)    
        else:
            #  unconstrained problems should have fixed mu := 1 
            self.mu                          = 1
            update_penalty_parameter_fn = lambda varargin : self.penaltyParameterIsFixed(varargin)
            self.is_feasible_to_tol_fn       = lambda tvi, tve: True # unconstrained is always true
        
        self.feasible_to_tol = self.is_feasible_to_tol_fn(self.tvi,self.tve)                                   
        self.tv              = max(self.tvi,self.tve)
        self.tv_l1           = self.tvi_l1 + self.tve_l1
        self.tv_l1_grad      = self.tvi_l1_grad + self.tve_l1_grad
        self.p               = self.mu*self.f + self.tv_l1
        if isinstance(self.tv_l1_grad,int):
            self.p_grad          = self.mu*self.f_grad + self.tv_l1_grad
        else:
            self.p_grad          = self.mu*self.f_grad + self.tv_l1_grad.reshape(self.f_grad.shape)
        
        #  to be able to rollback to a previous iterate and mu, specifically
        #  last time snapShot() was invoked
        self.fn_evals        = 1
        self.snap_shot       = None
        self.at_snap_shot    = False
        self.stat_value      = float("nan")

        if constrained: 
            self.unscale_fields_fn   = lambda data: self.unscaleFieldsConstrained(data)
            self.update_best_fn      = lambda : self.updateBestSoFarConstrained()
            get_best_fn         = lambda : self.getBestConstrained()
            self.most_feasible       = self.getInfoForXConstrained()
            self.best_to_tol         = None
        else:
            self.unscale_fields_fn   = lambda data : self.unscaleFields(data)
            self.update_best_fn      = lambda : self.updateBestSoFar()
            get_best_fn         = lambda : self.getBest()
            self.best_unconstrained  = self.getInfoForX()
        
        self.update_best_fn()

        # output object with methods
        penalty_fn_object = GeneralStruct()
        setattr(penalty_fn_object,"evaluatePenaltyFunction4linesearch",lambda x_in: self.evaluateAtX4linesearch(x_in))
        setattr(penalty_fn_object,"evaluatePenaltyFunction",lambda x_in: self.evaluateAtX(x_in))
        setattr(penalty_fn_object,"updatePenaltyParameter",update_penalty_parameter_fn)
        setattr(penalty_fn_object,"getX",lambda : self.getX())
        setattr(penalty_fn_object,"getComponentValues",lambda : self.getComponentValues())
        setattr(penalty_fn_object,"getPenaltyFunctionValue",lambda : self.getPenaltyFunctionValue())
        setattr(penalty_fn_object,"getPenaltyParameter",lambda : self.getPenaltyParameter())
        setattr(penalty_fn_object,"getGradients",lambda : self.getGradients())
        setattr(penalty_fn_object,"getNumberOfEvaluations",lambda : self.getNumberOfEvaluations())
        setattr(penalty_fn_object,"getBestSolutions",get_best_fn)
        setattr(penalty_fn_object,"isFeasibleToTol",lambda : self.isFeasibleToTolerances())
        setattr(penalty_fn_object,"isPrescaled",lambda : self.isPrescaled())
        setattr(penalty_fn_object,"hasConstraints",lambda : constrained)
        setattr(penalty_fn_object,"numberOfInequalities",lambda : self.ci.shape[0])
        setattr(penalty_fn_object,"numberOfEqualities",lambda : self.ce.shape[0])
        setattr(penalty_fn_object,"snapShot",lambda : self.snapShot())
        setattr(penalty_fn_object,"restoreSnapShot",lambda user_snap_shot = None: self.restoreSnapShot(user_snap_shot))
        setattr(penalty_fn_object,"addStationarityMeasure",lambda stationarity_measure: self.addStationarityMeasure(stationarity_measure))

        return [penalty_fn_object, grad_norms_at_x0]

    # evaluate objective, constraints, violation, and penalty function at x
    def evaluateAtX4linesearch(self,x_in):
        try: 
            self.at_snap_shot    = False
            self.stat_value      = float("nan")
            self.fn_evals        += 1
            # evaluate objective and its gradient
            self.f      = self.f_eval_fn(x_in)
            # evaluate constraints and their violations (nested update)
            self.eval_ineq_fn(x_in) 
            self.eval_eq_fn(x_in)
        except Exception as e:
            print("NCVX userSuppliedFunctionsError: failed to evaluate objective/constraint functions at x for line search.")
            print(traceback.format_exc())
            sys.exit()

        self.x                   = x_in
        self.feasible_to_tol     = self.is_feasible_to_tol_fn(self.tvi,self.tve);  
        self.tv                  = np.maximum(self.tvi,self.tve)
        self.tv_l1               = self.tvi_l1 + self.tve_l1
        # self.tv_l1_grad          = self.tvi_l1_grad + self.tve_l1_grad
        self.p                   = self.mu*self.f + self.tv_l1
        
        # # update best points encountered so far
        # self.update_best_fn()
        
        # copy nested variables values to output arguments
        p_out               = self.p
        feasible_to_tol_out = self.feasible_to_tol

        return [p_out,feasible_to_tol_out]

    # evaluate objective, constraints, violation, and penalty function at x
    def evaluateAtX(self,x_in):
        try: 
            self.at_snap_shot    = False
            self.stat_value      = float("nan")
            self.fn_evals        += 1
            # evaluate objective and its gradient
            [self.f,self.f_grad]      = self.obj_fn(x_in)
            # evaluate constraints and their violations (nested update)
            self.eval_ineq_fn(x_in) 
            self.eval_eq_fn(x_in)
        except Exception as e:
            print("NCVX userSuppliedFunctionsError: failed to evaluate objective/constraint functions at x.")
            print(traceback.format_exc())
            sys.exit()

        self.x                   = x_in
        self.feasible_to_tol     = self.is_feasible_to_tol_fn(self.tvi,self.tve);  
        self.tv                  = np.maximum(self.tvi,self.tve)
        self.tv_l1               = self.tvi_l1 + self.tve_l1
        self.tv_l1_grad          = self.tvi_l1_grad + self.tve_l1_grad
        self.p                   = self.mu*self.f + self.tv_l1
        if isinstance(self.tv_l1_grad,int):
            self.p_grad              = self.mu*self.f_grad + self.tv_l1_grad
        else:
            self.p_grad              = self.mu*self.f_grad + self.tv_l1_grad.reshape(self.f_grad.shape)
        
        # update best points encountered so far
        self.update_best_fn()
        
        # copy nested variables values to output arguments
        p_out               = self.p
        p_grad_out          = self.p_grad
        feasible_to_tol_out = self.feasible_to_tol

        return [p_out,p_grad_out,feasible_to_tol_out]

    def getComponentValues(self):
        f_o             = self.f
        f_grad_o        = self.f_grad
        tv_l1_o         = self.tv_l1
        tv_l1_grad_o    = self.tv_l1_grad
        return [f_o,f_grad_o,tv_l1_o,tv_l1_grad_o]

    def isFeasibleToTolerances(self):
        tf              = self.feasible_to_tol
        return tf

    def getNumberOfEvaluations(self):
        evals           = self.fn_evals
        return evals
   
    # return the most recently evaluated x
    def getX(self):
        x_out           = self.x
        return x_out

    def getPenaltyFunctionValue(self):
        p_out           = self.p
        p_grad_out      = self.p_grad
        return [p_out,p_grad_out]

    # update penalty function with new penalty parameter
    def updatePenaltyParameter(self,mu_new):
        self.mu              = mu_new
        self.p               = self.mu*self.f + self.tv_l1
        self.p_grad          = self.mu*self.f_grad + self.tv_l1_grad.reshape(self.f_grad.shape)
        p_new           = self.p
        p_grad_new      = self.p_grad;   
        return [p_new,p_grad_new,mu_new] 

    # for unconstrained problems, ignore updates to mu
    def penaltyParameterIsFixed(self,varargin):
        mu_new          = self.mu
        p_new           = self.p
        p_grad_new      = self.p_grad     
        return [p_new,p_grad_new,mu_new]

    def getPenaltyParameter(self):
        mu_out          = self.mu
        return mu_out

    def getGradients(self):
        f_grad_out      = self.f_grad
        ci_grad_out     = self.ci_grad
        ce_grad_out     = self.ce_grad
        return [f_grad_out, ci_grad_out, ce_grad_out]

    def isPrescaled(self):
        tf              = self.prescaled
        return tf

    def addStationarityMeasure(self,stationarity_measure):
        self.stat_value = stationarity_measure
        if self.at_snap_shot:
            self.snap_shot.stat_value = stationarity_measure
        
    ################## PRIVATE helper functions  ##################
       
    def snapShot(self):
        # scalings never change so no need to snapshot them
        self.snap_shot = GeneralStruct()
        setattr(self.snap_shot,"f",self.f)
        setattr(self.snap_shot,"f_grad",self.f_grad)
        setattr(self.snap_shot,"ci",self.ci)
        setattr(self.snap_shot,"ci_grad",self.ci_grad)
        setattr(self.snap_shot,"ce",self.ce)
        setattr(self.snap_shot,"ce_grad",self.ce_grad)
        setattr(self.snap_shot,"tvi",self.tvi)
        setattr(self.snap_shot,"tve",self.tve)
        setattr(self.snap_shot,"tv",self.tv)
        setattr(self.snap_shot,"tvi_l1",self.tvi_l1)
        setattr(self.snap_shot,"tvi_l1_grad",self.tvi_l1_grad)
        setattr(self.snap_shot,"tve_l1",self.tve_l1)
        setattr(self.snap_shot,"tve_l1_grad",self.tve_l1_grad)
        setattr(self.snap_shot,"tv_l1",self.tv_l1)
        setattr(self.snap_shot,"tv_l1_grad",self.tv_l1_grad)
        setattr(self.snap_shot,"p",self.p)
        setattr(self.snap_shot,"p_grad",self.p_grad)
        setattr(self.snap_shot,"mu",self.mu)
        setattr(self.snap_shot,"x",self.x)
        setattr(self.snap_shot,"feasible_to_tol",self.feasible_to_tol)
        setattr(self.snap_shot,"stat_value",self.stat_value)
       
        s = self.snap_shot
        self.at_snap_shot = True
        return s

    def restoreSnapShot(self, user_snap_shot = None):
        if user_snap_shot != None:
            s = user_snap_shot
        else:
            s = self.snap_shot
        
        if np.any(s != None):
            self.f               = s.f
            self.f_grad          = s.f_grad
            self.ci              = s.ci
            self.ci_grad         = s.ci_grad
            self.ce              = s.ce
            self.ce_grad         = s.ce_grad
            self.tvi             = s.tvi
            self.tve             = s.tve
            self.tv              = s.tv
            self.tvi_l1          = s.tvi_l1
            self.tvi_l1_grad     = s.tvi_l1_grad
            self.tve_l1          = s.tve_l1
            self.tve_l1_grad     = s.tve_l1_grad
            self.tv_l1           = s.tv_l1
            self.tv_l1_grad      = s.tv_l1_grad
            self.p               = s.p
            self.p_grad          = s.p_grad
            self.mu              = s.mu
            self.x               = s.x
            self.feasible_to_tol = s.feasible_to_tol
            self.stat_value      = s.stat_value
            self.snap_shot       = s
            self.at_snap_shot    = True
        
        return s
   

    def evalInequality(self,x,fn):
        [self.ci,self.ci_grad]                = fn(x)
        [self.tvi,self.tvi_l1,self.tvi_l1_grad]    = totalViolationInequality(self.ci,self.ci_grad)
        return

    def evalEquality(self,x,fn):
        [self.ce,self.ce_grad]                = fn(x)
        [self.tve,self.tve_l1,self.tve_l1_grad]    = totalViolationEquality(self.ce,self.ce_grad)
        return

    def isFeasibleToTol(self,tvi,tve):
        #  need <= since tolerances could be 0 for very demanding users ;-)
        tf = (tvi <= self.viol_ineq_tol and tve <= self.viol_eq_tol)
        return tf

    def getInfoForX(self):
        s = dataStruct(self.x,self.f)   
        return s

    def getInfoForXConstrained(self):   
        s = dataStructConstrained(self.x,self.f,self.ci,self.ce,self.tvi,self.tve,self.tv,self.feasible_to_tol,self.mu)
        return s

    def updateBestSoFar(self):
        if self.f < self.best_unconstrained.f:
            self.best_unconstrained = self.getInfoForX()
        

    def updateBestSoFarConstrained(self):      
        # Update the iterate which is closest to the feasible region.  In
        # the case of ties, keep the one that most minimizes the objective.
        update_mf   =   self.tv < self.most_feasible.tv or (self.tv == self.most_feasible.tv and self.f < self.most_feasible.f)
        
        # Update iterate which is feasible w.r.t violation tolerances and
        # most minimizes the objective function
        update_btt  =   self.feasible_to_tol and (np.all(self.best_to_tol == None) or self.f < self.best_to_tol.f)
        
        if update_mf or update_btt:
            soln = self.getInfoForXConstrained()
            if update_mf:
                self.most_feasible   = soln
            if update_btt:
                self.best_to_tol     = soln
            

    def unscaleFields(self,data):
        unscaled    = dataStruct(data.x,unscaleValues(data.f,self.scaling_f))
        return unscaled

    def unscaleFieldsConstrained(self,data):
        f_u         = unscaleValues(data.f,self.scaling_f)
        ci_u        = unscaleValues(data.ci,self.scaling_ci)
        ce_u        = unscaleValues(data.ce,self.scaling_ce)
        tvi_u       = totalViolationMax(violationsInequality(ci_u))
        tve_u       = totalViolationMax(violationsEquality(ce_u))
        tv_u        = np.max(tvi_u,tve_u)
        unscaled    = dataStructConstrained(data.x, f_u, ci_u, ce_u, tvi_u, tve_u, tv_u,self.is_feasible_to_tol_fn(tvi_u,tve_u), data.mu )
        return unscaled

    def getBest(self):
        final_field         = ('final', self.getInfoForX())
        best_field          = ('best', self.best_unconstrained)
        if self.prescaled:
            scalings_field  = self.getScalings()
            final_unscaled  = self.getUnscaledData(final_field)
            best_unscaled   = self.getUnscaledData(best_field)
        else:
            scalings_field  = ()
            final_unscaled  = ()
            best_unscaled   = ()
        
        soln = GeneralStruct()
        if scalings_field != ():
            setattr(soln,scalings_field[0],scalings_field[1])
        if final_field != ():
            setattr(soln,final_field[0],final_field[1])
        if final_unscaled != ():
            setattr(soln,final_unscaled[0],final_unscaled[1])
        if best_field != ():
            setattr(soln,best_field[0],best_field[1])
        if best_unscaled != ():
            setattr(soln,best_unscaled[0],best_unscaled[1])

        stat_value_o = self.stat_value
        return [soln, stat_value_o]

    def getBestConstrained(self):
        final_field         = ("final", self.getInfoForXConstrained())
        feas_field          = ("most_feasible", self.most_feasible)
        if np.all(self.best_to_tol == None):
            best_field      = ()
        else:
            best_field      = ("best" , self.best_to_tol)           
        
        if self.prescaled:
            scalings_field  = self.getScalings()
            final_unscaled  = self.getUnscaledData(final_field)
            feas_unscaled   = self.getUnscaledData(feas_field)
            best_unscaled   = self.getUnscaledData(best_field)
        else:
            scalings_field  = ()
            final_unscaled  = ()
            feas_unscaled   = ()
            best_unscaled   = ()
        
        soln = GeneralStruct()
        if scalings_field != ():
            setattr(soln,scalings_field[0],scalings_field[1])
        if final_field != ():
            setattr(soln,final_field[0],final_field[1])
        if final_unscaled != ():
            setattr(soln,final_unscaled[0],final_unscaled[1])
        if best_field != ():
            setattr(soln,best_field[0],best_field[1])
        if best_unscaled != ():
            setattr(soln,best_unscaled[0],best_unscaled[1])
        if feas_field != ():
            setattr(soln,feas_field[0],feas_field[1])
        if feas_unscaled != ():
            setattr(soln,feas_unscaled[0],feas_unscaled[1])
        
        stat_value_o = self.stat_value
        return [soln, stat_value_o]

    def getScalings(self):
        scalings_field      = ("scalings",self.scalings)
        return scalings_field

    def getUnscaledData(self,data_field):
        if data_field == ():
            unscaled_data   = ()
        else:
            name = data_field[0]
            data = data_field[1]
            unscaled_data   = ([name+"_unscaled"],self.unscale_fields_fn(data))
        return unscaled_data

def assertFnOutputs(n,f,g,fn_name):
    if fn_name == "objective":
        arg1 = "function value"
        arg2 = "gradient"
        assertFn(np.isscalar(f),arg1,fn_name,'be a scalar')
        [r,c] = g.shape
        assertFn(c == 1,arg2,fn_name,'be a column vector')
    else:
        arg1 = "function value(s)"
        arg2 = "gradient(s)"
        [nf,c] = f.shape
        assertFn(nf >= 1 and c == 1,arg1,fn_name,'be a column vector')
        [r,ng] = g.shape
        assertFn(nf == ng,'number of gradients',fn_name,'should match the number of constraint function values' )

    assertFn(r == n,arg2,fn_name,'have dimension matching the number of variables')
    assertFn(torch.isreal(f) if torch.is_tensor(f) else np.isreal(f),arg1,fn_name,'should be real valued')
    assertFn(torch.isreal(g)==True,arg2,fn_name,'should be real valued')
    assertFn(torch.isfinite(f) if torch.is_tensor(f) else np.isfinite(f) ,arg1,fn_name,'should be finite valued')
    assertFn(torch.isfinite(g),arg2,fn_name,'should be finite valued')
    return

def assertFn(cond,arg_name,fn_name,msg):
    if torch.is_tensor(cond):
        assert torch.all(cond),("NCVX userSuppliedFunctionsError: The {} at x0 returned by the {} function should {}!".format(arg_name,fn_name,msg)  )   
    else:
        assert np.all(cond),("NCVX userSuppliedFunctionsError: The {} at x0 returned by the {} function should {}!".format(arg_name,fn_name,msg)  )                                 

class Class_splitEvalAtX:
    def __init__(self):
        pass

    def splitEvalAtX(self,eval_at_x_fn,x0):
        
        self.eval_at_x_fn = eval_at_x_fn
        [f,f_grad,self.ci,self.ci_grad,self.ce,self.ce_grad] = self.eval_at_x_fn(x0)
    
        obj_fn = lambda x : self.objective(x)
        ineq_fn = (lambda varargin: self.inequality(varargin) ) if ( torch.is_tensor(self.ci) ) else (None)
        eq_fn = (lambda varargin: self.equality(varargin) ) if (torch.is_tensor(self.ce)) else (None)
        
        return [f,f_grad,obj_fn,ineq_fn,eq_fn] 

    def objective(self,x):
        [f,g,self.ci,self.ci_grad,self.ce,self.ce_grad] = self.eval_at_x_fn(x)
        return [f,g]

    def inequality(self,varargin):
        c       = self.ci
        c_grad  = self.ci_grad
        return [c,c_grad]
    
    def equality(self,varargin):
        c       = self.ce
        c_grad  = self.ce_grad
        return [c,c_grad]

def rescaleObjective(x,fn,scaling):
    [f,g]   = fn(x)
    f       = f*scaling
    g       = g*scaling
    return [f,g]

def violationsInequality(ci):
    vi = ci.detach().clone()
    violated_indx = ci >= 0
    not_violated_indx = ~violated_indx
    vi[not_violated_indx] = 0
    return [vi,violated_indx]

def violationsEquality(ce):
    ve = abs(ce)
    violated_indx = (ce >= 0);   #indeed, this is all of them 
    return [ve,violated_indx]

def totalViolationMax(v):
    if torch.all(v==0):
        v_max = 0
    else:
        v_max = torch.max(v).item()
    return v_max

def totalViolationInequality(ci,ci_grad):
    [vi,indx] = violationsInequality(ci)
    #  l_inf penalty term for feasibility measure
    tvi = totalViolationMax(vi)
    #  l_1 penalty term for penalty function
    tvi_l1 = torch.sum(vi)
    # indx used for select certain cols
    tvi_l1_grad = torch.sum(ci_grad[:,indx[:,0]],1)
    return [tvi,tvi_l1,tvi_l1_grad]

def totalViolationEquality(ce,ce_grad):
    [ve,indx] = violationsEquality(ce)
    # l_inf penalty term for feasibility measure
    tve = totalViolationMax(ve)
    # l_1 penalty term for penalty function
    tve_l1 = torch.sum(ve)
    tve_l1_grad = torch.sum(ce_grad[:,indx[:,0]],1) - torch.sum(ce_grad[:,(~indx[:,0])],1)

    return [tve,tve_l1,tve_l1_grad]

def rescaleConstraint(x,fn,scalings):
    [f,g]   = fn(x)
    f       = np.multiply(f,scalings)
    g       = g @ np.diag(scalings)
    return [f,g]

def unscaleValues(values,scalars):
    if np.any(scalars != None):
        values = np.divide(values,scalars)  
    return values

def setupConstraint( x0, c_fn, eval_fn, inequality_constraint, prescaling_threshold, torch_device):
    n = len(x0)            
    #  eval_fn is either a function handle for evaluateInequality or
    #  evaluateEquality so we can detect which we have based on its length
    if inequality_constraint:
        viol_fn = lambda ci, ci_grad : totalViolationInequality(ci,ci_grad)
        type_str = "in"
    else:
        viol_fn = lambda ce, ce_grad: totalViolationEquality(ce,ce_grad)
        type_str = ""

    scalings= None; # default if no prescaling is applied
    # isempty function
    if np.all( c_fn==None ):
        eval_fn_ret             = lambda x : None
        # These must have the right dimensions for computations to be 
        # done even if there are no such constraints
        c                   = torch.zeros((0,1),device=torch_device, dtype=torch.double)
        c_grad              = torch.zeros((len(x0),0),device=torch_device, dtype=torch.double)
        c_grad_norms        = 0
        tv                  = 0
        tv_l1               = 0
        tv_l1_grad          = 0
        constrained         = False
    elif isinstance(c_fn, types.LambdaType):
        try: 
            [c,c_grad]      = c_fn(x0)
        except Exception as e:
            print("NCVX userSuppliedFunctionsError : failed to evaluate [c,c_grad] = {}eq_fn(x0).".format(type_str))
            print(traceback.format_exc())
            sys.exit()

        assertFnOutputs(n,c,c_grad,type_str+"equality constraints") 
        c_grad_norms        = torch.sqrt(torch.sum(torch.square(c_grad),0)) 
        # indices of gradients whose norms are larger than limit
        indx                = c_grad_norms > prescaling_threshold
        if torch.any(indx !=0 ):
            scalings        = np.ones(len(c),1)
            # we want to rescale these "too large" functions so that 
            # the norms of their gradients are set to limit at x0
            scalings[indx]  = np.divide(prescaling_threshold, c_grad_norms(indx))
            c_fn            = lambda x: rescaleConstraint(x,c_fn,scalings)
            # rescale already computed constraints and gradients
            c               = np.multiply(c,scalings)
            c_grad          = c_grad @ np.diag(scalings)
        
        [tv,tv_l1,tv_l1_grad]   = viol_fn(c,c_grad)
        # reset eval_fn so that it computes the values and gradients 
        # for both the constraint and the corresponding violation
        eval_fn_ret             = lambda x: eval_fn(x,c_fn)
        constrained         = True
    else:       
        print("NCVX userSuppliedFunctionsError: {}eq_fn must be a function handle of x or empty, that is, None.\n".format(type_str))
    
    return [eval_fn_ret, c, c_grad, tv, tv_l1, tv_l1_grad, c_grad_norms, scalings, constrained]

def dataStruct(x,f):
    s = GeneralStruct()
    setattr(s,"x",x)
    setattr(s,"f",f)
    return s

def dataStructConstrained(x,f,ci,ce,tvi,tve,tv,feasible_to_tol,mu):
    s = GeneralStruct()
    setattr(s,"x",x)
    setattr(s,"f",f)
    setattr(s,"ci",ci)
    setattr(s,"ce",ce)
    setattr(s,"tvi",tvi)
    setattr(s,"tve",tve) 
    setattr(s,"tv",tv) 
    setattr(s,"feasible_to_tol",feasible_to_tol) 
    setattr(s,"mu",mu) 
    return s
    
