from numpy.core.numeric import Inf
import torch
from private import ncvxConstants as pgC
from private.optionValidator import oV
import numpy as np
from ncvxStruct import Options
from private.isAnInteger import isAnInteger
import traceback,sys

def ncvxOptions(n,options, torch_device):
    """
    ncvxOptions:
        Validate user options struct for ncvx.py.  If user_opts is None or
        not provided, returned opts will be NCVX's default parameters.
        Standard or advanced options may be set.  

       Type:
       help(ncvxOptionsAdvanced) 
       to see documentation for the advanced user options.
   
       USAGE:
       opts = ncvxOptions(n,options, torch_device)

       INPUT:
       n           Number of variables being optimized.

       user_opts   Struct of settable algorithm parameters.  No fields are 
                   required, irrelevant fields are ignored, and user_opts 
                   may be given as None.

       OUTPUT:
       opts        Struct of all tunable user parameters for NCVX.
                   If a field is provided in user_opts, then the user's 
                   value is checked to whether or not it is a valid value, 
                   and if so, it is set in opts.  Otherwise, an error is 
                   thrown.  If a field is not provided in user_opts, opts
                   will contain the field with NCVX's default value.  

       STANDARD PARAMETERS 

        x0
        ----------------
        n by 1 double precision torch tensor. Default value: torch.randn(n,1).to(device=torch_device, dtype=torch.double)

        Initial starting point. One should pick x0 such that the objective
        and constraint functions are smooth at and about x0. If this is
        difficult to ascertain, it is generally recommended to initialize
        NCVX at randomly-generated starting points.

        mu0
        ----------------
        Positive real value. Default value: 1

        Initial value of the penalty parameter. 
        NOTE: irrelevant for unconstrained optimization problems.

        H0
        ----------------
        n by n double precision torch tensor. Default value: torch.eye(n,device=torch_device, dtype=torch.double) 

        Initial inverse Hessian approximation.  In full-memory mode, and 
        if opts.checkH0 is true, NCVX will numerically assert that this
        matrix is positive definite. In limited-memory mode, that is, if
        opts.limited_mem_size > 0, no numerical checks are done but this 
        matrix must be a sparse matrix.

        checkH0
        ----------------
        Boolean value. Default value: True

        By default, NCVX will check whether or not H0 is numerically
        positive definite (by checking whether or not cholesky() succeeds).
        However, when restarting NCVX from the last iterate of an earlier
        run, using soln.H_final (the last BFGS approximation to the inverse
        Hessian), soln.H_final may sometimes fail this check.  Set this
        option to False to disable it. No positive definite check is done
        when limited-memory mode is enabled.

        scaleH0
        ----------------
        Boolean value. Default value: True

        Scale H0 during BFGS/L-BFGS updates.  For full-memory BFGS, scaling
        is only applied on the first iteration only, and is generally only
        recommended when H0 is the identity (which is NCVX's default).
        For limited-memory BFGS, H0 is scaled on every update.  For more
        details, see opts.limited_mem_fixed_scaling.

        bfgs_damping
        ----------------
        Real value in [0,1]. Default value: 1e-4 
            
        This feature will adaptively damp potentially problematic BFGS
        updates in order to help ensure that the (L)BFGS inverse Hessian
        approximation always remains positive definite numerically.  The
        closer this value is to one, the more frequently and aggressively
        damping will be applied to (L)BFGS updates.  Setting this value to
        zero completely disables damping.

        limited_mem_size
        ----------------
        Non-negative integer. Default value: 0

        By default, NCVX uses full-memory BFGS updating.  For nonsmooth
        problems, full-memory BFGS is generally recommended.  However, if
        this is not feasible, one may optionally enable limited-memory BFGS
        updating by setting opts.limited_mem_size to a positive integer
        (significantly) less than the number of variables.

        limited_mem_fixed_scaling
        --------------------------------
        Boolean value. Default value: True

        In contrast to full-memory BFGS updating, limited-memory BFGS
        permits that H0 can be scaled on every iteration.  By default,
        NCVX will reuse the scaling parameter that is calculated on the
        very first iteration for all subsequent iterations as well.  Set
        this option to False to force NCVX to calculate a new scaling
        parameter on every iteration.  Note that opts.scaleH0 has no effect
        when opts.limited_mem_fixed_scaling is set to True.

        limited_mem_warm_start
        --------------------------------
        Python dictionary with key to be 'S', 'Y', 'rho' and 'gamma'. Default value: None
            
        If one is restarting NCVX, the previous L-BFGS information can be
        recycled by setting opts.limited_mem_warm_start = soln.H_final,
        where soln is NCVX's output struct from a previous run.  Note
        that one can either reuse the previous H0 or set a new one.

        prescaling_threshold
        --------------------------------
        Positive real value. Default value: Inf

        Pre-scales objective/constraint functions such that the norms of 
        their gradients evaluated at x0 do not exceed prescaling_threshold.  
        Inf (default) disables all pre-scaling.  Problems that are poorly
        scaled, that is, the gradients have norms that are large, may cause 
        difficulties for optimization.  Pre-scaling can help mitigate these 
        issues in an automatic way but, ideally, the user should consider 
        whether an alterative formulation of the optimization problem with 
        better inherent scaling is possible.  
        NOTE: solutions obtained for a pre-scaled problem MAY NOT BE a
        solutions for the originally specified problem.

        prescaling_info_msg
        --------------------------------
        Boolean value. Default value: True

        Prints a notice that NCVX has either automatically pre-scaled at
        least one of the objective or constraint functions or it has
        deteced that the optimization problem may be poorly scaled.  For
        more details, see opts.prescaling_threshold.  

        opt_tol     
        ----------------        
        Positive real value. Default value: 1e-8

        Tolerance for reaching (approximate) optimality/stationarity.
        See opts.ngrad, opts.evaldist, and the description of NCVX's 
        output argument soln, specifically the subsubfield .dnorm for more
        information.

        rel_tol
        ----------------
        Non-negative real value. Default value: 0

        Tolerance for determining when the relative decrease in the penalty
        function is sufficiently small.  NCVX will terminate if when 
        the relative decrease in the penalty function is at or below this
        tolerance and the current iterate is feasible to tolerances.
        Generally, we don't recommend using this feature since small steps
        are not necessarily indicative of being near a stationary point,
        particularly for nonsmooth problems.

        step_tol
        ----------------
        Positive real value. Default value: 1e-12

        Tolerance for how small of a step the line search will attempt
        before terminating.

        viol_ineq_tol
        ----------------                  
        Non-negative real value. Default value: 0

        Acceptable total violation tolerance of the inequality constraints.   

        viol_eq_tol
        ----------------                   
        Non-negative real value. Default value: 1e-6

        Acceptable total violation tolerance of the equality constraints. 

        ngrad
        ----------------
        Positive integer. Default value: min([100, 2*n, n+10])
                                
        Max number of previous gradients to be cached.  The QP defining 
        NCVX's measure of stationarity requires a history of previous 
        gradients.  Note that large values of ngrad can make the related QP
        expensive to solve, if a significant fraction of the currently
        cached gradients were evaluated at points within evaldist of the 
        current iterate.  Using 1 is recommended if and only if the problem 
        is unconstrained and the objective is known to be smooth.  See 
        opts.opt_tol, opts.evaldist, and the description of NCVX's output
        argument soln, specifically the subsubfield .dnorm for more
        information.

        evaldist
        ----------------                       
        Positive real value. Default value: 1e-4

        Previously evaluated gradients are only used in the stationarity 
        test if they were evaluated at points that are within distance 
        evaldist of the current iterate x.  See opts.opt_tol, opts.ngrad, 
        and the description of NCVX's output argument soln, specifically 
        the subsubfield .dnorm for more information.

        maxit
        ----------------
        Positive integer. Default value: 1000

        Max number of iterations.

        maxclocktime
        ----------------
        Positive real number Default value: Inf

        Quit if the elapsed clock time in seconds exceeds this.


        fvalquit
        ----------------
        Positive real value. Default value: -Inf

        Quit if objective function drops below this value at a feasible 
        iterate (that is, satisfying feasibility tolerances 
        opts.viol_ineq_tol and opts.viol_eq_tol).


        halt_on_linesearch_bracket     
        --------------------------------          
        Boolean value. Default value: True

        If the line search brackets a minimizer but fails to satisfy the 
        weak Wolfe conditions (necessary for a step to be accepted), NCVX 
        will terminate at this iterate when this option is set to true 
        (default).  For unconstrained nonsmooth problems, it has been 
        observed that this type of line search failure is often an 
        indication that a stationarity has in fact been reached.  By 
        setting this parameter to False, NCVX will instead first attempt 
        alternative optimization strategies (if available) to see if
        further progress can be made before terminating.   See
        gransoOptionsAdvanced for more details on NCVX's available 
        fallback optimization strategies and how they can be configured. 

        quadprog_info_msg
        --------------------------------
        Boolean value. Default value: True

        Prints a notice that NCVX's requires a quadprog-compatible QP
        solver and that the choice of QP solver may affect NCVX's quality
        of performance, in terms of efficiency and level of optimization. 


        print_level     
        ----------------
        Integer in {0,1}. Default value: 1

        Level of detail printed to console regarding optimization progress:

        0 - no printing whatsoever

        1 - prints info for each iteration  

        print_frequency      
        ----------------          
        Positive integer. Default value: 1

        Sets how often the iterations are printed.  When set to 1, every
        iteration is printed; when set to 10, only every 10th iteration is
        printed.  When set to Inf, no iterations are printed, except for
        at x0.  Note that this only affects .print_level == 1 printing;
        all messages from higher values of .print_level will still be
        printed no matter what iteration they occurred on.

        print_width  
        ----------------
        Integer in [9,23]. Default value: 14

        Number of characters wide to print values for the penalty function,
        the objective function, and the total violations of the inequality 
        and equality constraints. 

        print_print_ascii     
        --------------------------------          
        Boolean value. Default value: False

        By default, NCVX's printed output uses the extended character map, 
        so nice looking tables can be made.  But if you need to record the output, 
        you can restrict the printed output to only use the basic ASCII character map


        print_use_orange   
        --------------------------------
        Boolean value. Default value: True

        NCVX's uses orange
        printing to highlight pertinent information.  However, the user
        is the given option to disable it if they need to record the output

        halt_log_fn
        --------------------------------
        Lambda Function. Default value: None

        A user-provided function handle that is called on every iteration
        to allow the user to signal to NCVX for it to halt at that 
        iteration and/or create historical logs of the progress of the
        algorithm. 
        
        END OF STANDARD PARAMETERS

        See also ncvxOptions, ncvxOptionsAdvanced.

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
            granso.m introduced in GRANSO Version 1.0.
            
            Buyun Dec 20, 2021 (NCVX Version 1.0.0):
                ncvxOptions.py is translated from gransoOptions.m in GRANSO Version 1.6.4.

                Add new options:
                    QPsolver, init_step_size, linesearch_maxit, is_backtrack_linesearch,
                    searching_direction_rescaling, disable_terminationcode_6
                    See https://ncvx.org/settings/new_para.html for more details 


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
    

    #  Storing information in mememory
    default_opts = None
    LAST_FALLBACK_LEVEL = -1

    # This will be disabled by the default options or if the user does not
    # activate debug mode
    debug_mode = True
    
    #  need error handler here
    assert isinstance(n,int) and n > 0,'NCVX invalidUserOption: Number of variables n must be a positive integer!'
    

    if default_opts == None:
        [default_opts, LAST_FALLBACK_LEVEL] = getDefaults(n)

    if options == None:
        opts = postProcess(n,default_opts, torch_device)
        return opts
    else:
        user_opts = options
    
    #  need error handler here
    assert isinstance(user_opts, Options) ,'NCVX invalidUserOption: NCVX options must provided as an object of class Options!'

    # USER PROVIDED THEIR OWN OPTIONS SO WE MUST VALIDATE THEM
    validator_obj = oV()
    validator = validator_obj.optionValidator('NCVX',default_opts)
    validator.setUserOpts(user_opts)

    # surround the validation so we can rethrow the error from NCVX
    try: 
        #  Set debug mode first as we need its value in the catch block
        debug_mode = validator.getValue("debug_mode")
        validator.setLogical("debug_mode")
        
        #  SET INITIAL POINT AND PENALTY PARAMETER VALUE
        if hasattr(user_opts,"x0") and np.any(user_opts.x0 != None):
            validator.setColumnDimensioned("x0",n)
            validator.setRealFiniteValued("x0")
        
        validator.setRealNonnegative("mu0")
        
        #  SET INITIAL (L)BFGS DATA
        validator.setLogical("checkH0")
        validator.setLogical("scaleH0")
        validator.setRealInIntervalCC("bfgs_damping",0,1)
        validator.setLogical("limited_mem_fixed_scaling")
        validator.setIntegerNonnegative("limited_mem_size")
        lim_mem_size    = validator.getValue("limited_mem_size")
        lim_mem_mode    = lim_mem_size > 0
        if lim_mem_mode and hasattr(user_opts,"limited_mem_warm_start")  and user_opts.limited_mem_warm_start != None:
            
            #  Ensure all the necessary subfields for L-BFGS data exist and 
            #  if so, it returns a validator for this sub-struct of data.
            validator.setRestartData("limited_mem_warm_start")            
            ws              = user_opts.limited_mem_warm_start  
            [n_S,cols_S]    = ws['S'].shape
            [n_Y,cols_Y]    = ws['Y'].shape
            [_,cols_rho]    = ws['rho'].shape
            assert n == n_S and n == n_Y,'NCVX invalidUserOption: the number of rows in both subfields S and Y must match the number of optimization variables'
            assert cols_S > 0 and cols_S == cols_Y and cols_S == cols_rho,'NCVX invalidUserOption: subfields S, Y, and rho must all have the same (positive) number of columns'          
        
        if hasattr(user_opts,"H0") and torch.any(user_opts.H0) != None:
            validator.setDimensioned("H0",n,n)
            validator.setRealFiniteValued("H0")
            if lim_mem_mode:
                validator.setSparse("H0")
            elif validator.getValue("checkH0"):
                validator.setPositiveDefinite("H0")
        
        #  SET PRESCALING PARAMETERS
        validator.setRealPositive("prescaling_threshold")
        validator.setLogical("prescaling_info_msg")

        #  CONVERGE CRITERIA / PARAMETERS
        #  allow users to set zero optimality and violation tolerances, even 
        #  though it's a bit demanding ;-)
        validator.setRealNonnegative("opt_tol")
        validator.setRealNonnegative("rel_tol")
        validator.setRealPositive("step_tol")
        validator.setRealNonnegative("viol_ineq_tol")
        validator.setRealNonnegative("viol_eq_tol")
        validator.setIntegerPositive("ngrad")
        validator.setRealPositive("evaldist")

        #  EARLY TERMINATION PARAMETERS
        validator.setIntegerPositive("maxit")
        validator.setRealPositive("maxclocktime")
        validator.setReal("fvalquit")
        validator.setLogical("halt_on_quadprog_error")
        validator.setLogical("halt_on_linesearch_bracket")

        #  FALLBACK PARAMETERS (allowable last resort "heuristics")
        validator.setIntegerInRange("min_fallback_level", 0,LAST_FALLBACK_LEVEL)
        #  Use the custom validator so we can set a custom message
        validator.validateAndSet( "max_fallback_level", 
                                lambda x: isAnInteger(x) and x >= validator.getValue("min_fallback_level") and x <= LAST_FALLBACK_LEVEL,                          
                                "an integer in {opts.min_fallback_level,...,%d}" % LAST_FALLBACK_LEVEL)

        validator.setIntegerPositive("max_random_attempts")

        #  STEERING PARAMETERS
        validator.setLogical("steering_l1_model")
        validator.setRealNonnegative("steering_ineq_margin")
        validator.setIntegerPositive("steering_maxit")
        validator.setRealInIntervalOO("steering_c_viol",0,1)
        validator.setRealInIntervalOO("steering_c_mu",0,1) 
        validator.setLogical("quadprog_info_msg")
        validator.setString("QPsolver")
        validator.setRealInIntervalCC("regularize_threshold",1,np.inf)
        validator.setLogical("regularize_max_eigenvalues")

        #  LINE SEARCH PARAMETERS
        #  wolfe1: conventionally wolfe1 should be positive in (0,1) but
        #  zero is is usually fine in practice (though there are
        #  exceptions).  1 is not acceptable.
        #  wolfe2: conventionally wolfe2 should be > wolfe1 but it is
        #  sometimes okay for both to be zero (e.g. Shor)
        validator.setRealInIntervalCC("wolfe1",0,0.5); 
        validator.setRealInIntervalCO("wolfe2",validator.getValue('wolfe1'),1)                             
        validator.setIntegerNonnegative("linesearch_nondescent_maxit")
        validator.setIntegerNonnegative("linesearch_reattempts")
        validator.setIntegerNonnegative("linesearch_reattempts_x0")
        validator.setRealInIntervalOO("linesearch_c_mu",0,1)   
        validator.setRealInIntervalOO("linesearch_c_mu_x0",0,1)    

        validator.setIntegerNonnegative("linesearch_maxit")
        validator.setRealNonnegative("init_step_size")
        validator.setLogical("is_backtrack_linesearch")
        validator.setLogical("double_precision")
        validator.setLogical("searching_direction_rescaling")
        validator.setLogical("disable_terminationcode_6")        

        #  LOGGING PARAMETERS
        validator.setIntegerInRange("print_level",0,3)
        validator.setIntegerInRange("print_frequency",1,np.inf)
        validator.setIntegerInRange("print_width",9,23)
        validator.setLogical("print_ascii")
        validator.setLogical("print_use_orange")
                
        if hasattr(user_opts,"halt_log_fn") and user_opts.halt_log_fn != None:
            validator.setFunctionHandle("halt_log_fn")
        
        opts = validator.getValidatedOpts()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit()

    #  GET THE VALIDATED OPTIONS AND POST PROCESS THEM
    opts = postProcess(n,validator.getValidatedOpts(), torch_device)

    return opts

def postProcess(n,opts, torch_device):
    
    # bump up the max fallback level if necessary
    if opts.max_fallback_level < opts.min_fallback_level:
        opts.max_fallback_level = opts.max_fallback_level
    
    # If an initial starting point was not provided, use random vector
    if opts.double_precision:
        torch_dtype = torch.double
    else:
        torch_dtype = torch.float

    if opts.x0 == None:
        opts.x0 = torch.randn(n,1).to(device=torch_device, dtype=torch_dtype)
    
    # If an initial inverse Hessian was not provided, use the identity
    if opts.H0 == None:
        opts.H0 = torch.eye(n,device=torch_device, dtype=torch_dtype) 
    
    if hasattr(opts,"QPsolver"):
        QPsolver = opts.QPsolver
    
    return opts

def getDefaults(n):
    [*_, LAST_FALLBACK_LEVEL] = pgC.ncvxConstants()

    # default options for NCVX
    default_opts = Options()
    setattr(default_opts,'x0',None)
    setattr(default_opts,'mu0',1)
    setattr(default_opts,'H0',None)
    setattr(default_opts,'checkH0',True)
    setattr(default_opts,'scaleH0',True)
    setattr(default_opts,'bfgs_damping',1e-4)
    setattr(default_opts,'limited_mem_size',0)
    setattr(default_opts,'limited_mem_fixed_scaling',True)
    setattr(default_opts,'limited_mem_warm_start',None)
    setattr(default_opts,'prescaling_threshold',Inf)
    setattr(default_opts,'prescaling_info_msg',True)
    setattr(default_opts,'opt_tol',1e-8)
    setattr(default_opts,'rel_tol',0)
    setattr(default_opts,'step_tol',1e-12)
    setattr(default_opts,'viol_ineq_tol',1e-6)
    setattr(default_opts,'viol_eq_tol',1e-6)
    setattr(default_opts,'ngrad',min([100,2*n,n+10]))
    setattr(default_opts,'evaldist',1e-4)
    setattr(default_opts,'maxit',1000)
    setattr(default_opts,'maxclocktime',Inf)
    setattr(default_opts,'fvalquit',-Inf)
    setattr(default_opts,'halt_on_quadprog_error',False)
    setattr(default_opts,'halt_on_linesearch_bracket',True)
    setattr(default_opts,'min_fallback_level',0)
    setattr(default_opts,'max_fallback_level',LAST_FALLBACK_LEVEL-1)
    setattr(default_opts,'max_random_attempts',5)
    setattr(default_opts,'steering_l1_model',True)
    setattr(default_opts,'steering_ineq_margin',1e-6)
    setattr(default_opts,'steering_maxit',10)
    setattr(default_opts,'steering_c_viol',0.1)
    setattr(default_opts,'steering_c_mu',0.9)
    setattr(default_opts,'regularize_threshold',Inf)
    setattr(default_opts,'regularize_max_eigenvalues',False)
    setattr(default_opts,'quadprog_info_msg',True)
    setattr(default_opts,'QPsolver','osqp')
    setattr(default_opts,'wolfe1',1e-4)
    setattr(default_opts,'wolfe2',0.5)
    setattr(default_opts,'linesearch_nondescent_maxit',0)
    setattr(default_opts,'linesearch_reattempts',0)
    setattr(default_opts,'linesearch_reattempts_x0',10)
    setattr(default_opts,'linesearch_c_mu',0.5)
    setattr(default_opts,'linesearch_c_mu_x0',0.5)

    setattr(default_opts,'linesearch_maxit',np.inf)
    setattr(default_opts,'init_step_size',1)
    setattr(default_opts,'is_backtrack_linesearch',False)
    setattr(default_opts,'double_precision',True)
    setattr(default_opts,'searching_direction_rescaling',False)
    setattr(default_opts,'disable_terminationcode_6',False)

    setattr(default_opts,'print_level',1)
    setattr(default_opts,'print_frequency',1)
    setattr(default_opts,'print_width',14)
    setattr(default_opts,'print_ascii',False)
    setattr(default_opts,'print_use_orange',True)
    setattr(default_opts,'halt_log_fn',None)
    setattr(default_opts,'debug_mode',False)

    return [default_opts, LAST_FALLBACK_LEVEL]
    