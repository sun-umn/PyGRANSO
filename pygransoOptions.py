from numpy.core.numeric import Inf
from private import pygransoConstants as pgC
from private.optionValidator import oV
import numpy as np
from scipy import sparse
from pygransoStruct import Options
from dbg_print import dbg_print
from private.isAnInteger import isAnInteger
from numpy.random import default_rng

def gransoOptions(n,options):
    """
    gransoOptions:
        Validate user options struct for pygranso.py.  If user_opts is [] or
        not provided, returned opts will be PyGRANSO's default parameters.
        Standard or advanced options may be set.  

       Type:
       >> help(gransoOptionsAdvanced) 
       to see documentation for the advanced user options.
   
       USAGE:
       opts = gransoOptions(n);
       opts = gransoOptions(n,user_opts);

       INPUT:
       n           Number of variables being optimized.

       user_opts   Struct of settable algorithm parameters.  No fields are 
                   required, irrelevant fields are ignored, and user_opts 
                   may be given as None.

       OUTPUT:
       opts        Struct of all tunable user parameters for PyGRANSO.
                   If a field is provided in user_opts, then the user's 
                   value is checked to whether or not it is a valid value, 
                   and if so, it is set in opts.  Otherwise, an error is 
                   thrown.  If a field is not provided in user_opts, opts
                   will contain the field with PyGRANSO's default value.  

       STANDARD PARAMETERS 

        .x0                             [n by 1 real vector | {randn(n,1)}]
       Initial starting point.  One should pick x0 such that the objective
       and constraint functions are smooth at and about x0.  If this is
       difficult to ascertain, it is generally recommended to initialize
       PyGRANSO at randomly-generated starting points.

       .mu0                            [real > 0 | {1}]
       Initial value of the penalty parameter.
       NOTE: irrelevant for unconstrained optimization problems.

       .H0:                            [n by n real matrix | {speye(n)}]
       Initial inverse Hessian approximation.  In full-memory mode, and 
       if opts.checkH0 is true, PyGRANSO will numerically assert that this
       matrix is positive definite.  In limited-memory mode, that is, if
       opts.limited_mem_size > 0, no numerical checks are done but this 
       matrix must be a sparse matrix.
   
       .checkH0                        [logical | {true}]
       By default, PyGRANSO will check whether or not H0 is numerically
       positive definite (by checking whether or not chol() succeeds).
       However, when restarting PyGRANSO from the last iterate of an earlier
       run, using soln.H_final (the last BFGS approximation to the inverse
       Hessian), soln.H_final may sometimes fail this check.  Set this
       option to false to disable it.  No positive definite check is done
       when limited-memory mode is enabled.

      .scaleH0                        [logical | {true}]
       Scale H0 during BFGS/L-BFGS updates.  For full-memory BFGS, scaling
       is only applied on the first iteration only, and is generally only
       recommended when H0 is the identity (which is PyGRANSO's default).
       For limited-memory BFGS, H0 is scaled on every update.  For more
       details, see opts.limited_mem_fixed_scaling.
 
       .bfgs_damping                   [real in [0,1] | {1e-4}]
       This feature will adaptively damp potentially problematic BFGS
       updates in order to help ensure that the (L)BFGS inverse Hessian
       approximation always remains positive definite numerically.  The
       closer this value is to one, the more frequently and aggressively
       damping will be applied to (L)BFGS updates.  Setting this value to
       zero completely disables damping.

        .limited_mem_size               [nonnegative integer | {0}]
       By default, PyGRANSO uses full-memory BFGS updating.  For nonsmooth
       problems, full-memory BFGS is generally recommended.  However, if
       this is not feasible, one may optionally enable limited-memory BFGS
       updating by setting opts.limited_mem_size to a positive integer
       (significantly) less than the number of variables.
  
        .limited_mem_fixed_scaling      [logical | {true}]
       In contrast to full-memory BFGS updating, limited-memory BFGS
       permits that H0 can be scaled on every iteration.  By default,
       PyGRANSO will reuse the scaling parameter that is calculated on the
       very first iteration for all subsequent iterations as well.  Set
       this option to false to force PyGRANSO to calculate a new scaling
       parameter on every iteration.  Note that opts.scaleH0 has no effect
       when opts.limited_mem_fixed_scaling is set to true.

       .limited_mem_warm_start         [struct | {[]}]
       If one is restarting PyGRANSO, the previous L-BFGS information can be
       recycled by setting opts.limited_mem_warm_start = soln.H_final,
       where soln is PyGRANSO's output struct from a previous run.  Note
       that one can either reuse the previous H0 or set a new one.
       
       .prescaling_threshold           [real > 0 | {inf}]
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

       .prescaling_info_msg            [logical | {true}]
       Prints a notice that PyGRANSO has either automatically pre-scaled at
       least one of the objective or constraint functions or it has
       deteced that the optimization problem may be poorly scaled.  For
       more details, see opts.prescaling_threshold.  

       .opt_tol                        [real >= 0 | {1e-8}]
       Tolerance for reaching (approximate) optimality/stationarity.
       See opts.ngrad, opts.evaldist, and the description of PyGRANSO's 
       output argument soln, specifically the subsubfield .dnorm for more
       information.

       .rel_tol                        [real >= 0 | {0}]
       Tolerance for determining when the relative decrease in the penalty
       function is sufficiently small.  PyGRANSO will terminate if when 
       the relative decrease in the penalty function is at or below this
       tolerance and the current iterate is feasible to tolerances.
       Generally, we don't recommend using this feature since small steps
       are not necessarily indicative of being near a stationary point,
       particularly for nonsmooth problems.

       .step_tol                       [real > 0 | {1e-12}]
       Tolerance for how small of a step the line search will attempt
       before terminating.

       .viol_ineq_tol                  [real >= 0 | {0}]
       Acceptable total violation tolerance of the inequality constraints.     

       .viol_eq_tol                    [real >= 0 | {1e-6}]
       Acceptable total violation tolerance of the equality constraints.          

       .ngrad                          [integer > 0 | {min([100, 2*n, n+10])}]
       Max number of previous gradients to be cached.  The QP defining 
       PyGRANSO's measure of stationarity requires a history of previous 
       gradients.  Note that large values of ngrad can make the related QP
       expensive to solve, if a significant fraction of the currently
       cached gradients were evaluated at points within evaldist of the 
       current iterate.  Using 1 is recommended if and only if the problem 
       is unconstrained and the objective is known to be smooth.  See 
       opts.opt_tol, opts.evaldist, and the description of PyGRANSO's output
       argument soln, specifically the subsubfield .dnorm for more
       information.

       .evaldist                       [real > 0 | {1e-4}]
       Previously evaluated gradients are only used in the stationarity 
       test if they were evaluated at points that are within distance 
       evaldist of the current iterate x.  See opts.opt_tol, opts.ngrad, 
       and the description of PyGRANSO's output argument soln, specifically 
       the subsubfield .dnorm for more information.

       .maxit                          [integer > 0 | {1000}]
       Max number of iterations.

       .maxclocktime                   [real > 0 | {inf}] 
       Quit if the elapsed clock time in seconds exceeds this.
                   
       .fvalquit                       [any real | {-inf}] 
       Quit if objective function drops below this value at a feasible 
       iterate (that is, satisfying feasibility tolerances 
       opts.viol_ineq_tol and opts.viol_eq_tol).
 
       .halt_on_quadprog_error         [logical | {false}]
       By default, PyGRANSO will attempt to 'work around' any quadprog
       failure (numerically invalid result or quadprog throws a bonafide
       error) according to a set of default fallback strategies (see
       gransoOptionsAdvanced for how these can be configured).  Generally,
       users should expect quadprog to mostly work, with errors only
       occurring quite rarely.  However, if quadprog fails frequently,
       then PyGRANSO's performance will likely be greatly hindered (in terms
       of efficiency and quality of optimization).  Set this option to
       true if one wishes PyGRANSO to halt on the first quadprog error 
       encountered while computing the search direction.

       .halt_on_linesearch_bracket     [logical | {true}]
       If the line search brackets a minimizer but fails to satisfy the 
       weak Wolfe conditions (necessary for a step to be accepted), PyGRANSO 
       will terminate at this iterate when this option is set to true 
       (default).  For unconstrained nonsmooth problems, it has been 
       observed that this type of line search failure is often an 
       indication that a stationarity has in fact been reached.  By 
       setting this parameter to false, PyGRANSO will instead first attempt 
       alternative optimization strategies (if available) to see if
       further progress can be made before terminating.   See
       gransoOptionsAdvanced for more details on PyGRANSO's available 
       fallback optimization strategies and how they can be configured.

       .quadprog_info_msg              [logical | {true}]
       Prints a notice that PyGRANSO's requires a quadprog-compatible QP
       solver and that the choice of QP solver may affect PyGRANSO's quality
       of performance, in terms of efficiency and level of optimization. 

       .print_level                    [integer in {0,1,2,3} | 1]
       Level of detail printed to console regarding optimization progress:
           0 - no printing whatsoever
           1 - prints info for each iteration  
           2 - adds additional info about BFGS updates and line searches
           3 - adds info on any errors that are encountered

       .print_frequency                [integer in {1,2,3,...,inf} | 1]
       Sets how often the iterations are printed.  When set to one, every
       iteration is printed; when set to 10, only every 10th iteration is
       printed.  When set to inf, no iterations are printed, except for
       at x0.  Note that this only affects .print_level == 1 printing;
       all messages from higher values of .print_level will still be
       printed no matter what iteration they occurred on.

       .print_width                    [integer in {9,...,23} | {14}]
       Number of characters wide to print values for the penalty function,
       the objective function, and the total violations of the inequality 
       and equality constraints. 

       .print_ascii                    [logical | {false}]
       By default, PyGRANSO's printed output uses the extended character
       map, so nice looking tables can be made.  However, diary() does not
       capture these symbols.  So, if you need to record the output, you
       can restrict the printed output to only use the basic ASCII
       character map, which may look better when captured by diary().

       .print_use_orange               [logical | {true}]
       By default, PyGRANSO's printed output makes use of an undocumented
       MATLAB feature for printing orange text.  PyGRANSO's uses orange
       printing to highlight pertinent information.  However, the user
       is the given option to disable it, since support cannot be
       guaranteed (since it is an undocumented feature).
  
       .halt_log_fn                    [a function handle | {[]}]  
       A user-provided function handle that is called on every iteration
       to allow the user to signal to PyGRANSO for it to halt at that 
       iteration and/or create historical logs of the progress of the
       algorithm.  For more details, see also makeHaltLogFunctions in the
       halt_log_template folder, which shows the function signature
       halt_log_fn must have if supplied.

       .debug_mode                     [logical | {false}]
       By default, PyGRANSO will catch any errors that occur during runtime,
       in order to be able to return the best computed result so far. 
       Instead of rethrowing the error, PyGRANSO will instead print an error
       message without and add the error object to PyGRANSO's struct output
       argument soln.  However, this behavior can make it harder to debug
       PyGRANSO so it can be disabled by setting this option to true.
        
        END OF STANDARD PARAMETERS

       See also pygranso, pygransoOptionsAdvanced, and makeHaltLogFunctions.

    """
    
    #  Storing information in mememory
    # persistent default_opts;
    # persistent LAST_FALLBACK_LEVEL;
    default_opts = None
    LAST_FALLBACK_LEVEL = -1

    # This will be disabled by the default options or if the user does not
    # activate debug mode
    debug_mode = True
    
    #  need error handler here
    assert isinstance(n,int) and n > 0,'PyGRANSO invalidUserOption: Number of variables n must be a positive integer!'
    

    if default_opts == None:
        [default_opts, LAST_FALLBACK_LEVEL] = getDefaults(n)

    if options == None:
        opts = postProcess(n,default_opts)
        return opts
    else:
        user_opts = options
    
    #  need error handler here
    assert isinstance(user_opts, Options) ,'PyGRANSO invalidUserOption: PyGRANSO options must provided as an object of class Options!'

    # USER PROVIDED THEIR OWN OPTIONS SO WE MUST VALIDATE THEM
    validator_obj = oV()
    validator = validator_obj.optionValidator('PyGRANSO',default_opts)
    validator.setUserOpts(user_opts)

    # surround the validation so we can rethrow the error from PyGRANSO
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
            lbfgs_validator = validator.setStructWithFields( "limited_mem_warm_start","S","Y","rho","gamma")
            
            ws              = user_opts.limited_mem_warm_start
            [n_S,cols_S]    = ws.S.shape
            [n_Y,cols_Y]    = ws.Y.shape
            cols_rho        = ws.rho.shape[1]
            
            dbg_print("skip pygransoOptions lbfgs_validator.assert")
            # lbfgs_validator.assert(                                     ...
            #     n == n_S && n == n_Y,                                   ...
            #     [   'the number of rows in both subfields S and Y must '...
            #         'match the number of optimization variables'        ]);
            # lbfgs_validator.assert(                                     ...
            #     cols_S > 0 && cols_S == cols_Y && cols_S == cols_rho,   ...
            #     [   'subfields S, Y, and rho must all have the same '   ...
            #         '(positive) number of columns'                      ]);
            
            lbfgs_validator.setRow("rho")            
            
            lbfgs_validator.setRealFiniteValued("S")
            lbfgs_validator.setRealFiniteValued("Y")
            lbfgs_validator.setRealFiniteValued("rho")
            lbfgs_validator.setReal("gamma")
            lbfgs_validator.setFiniteValued("gamma")
        
        if hasattr(user_opts,"H0") and user_opts.H0 != None:
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
        validator.setRealInIntervalCO("wolfe2",validator.getValue('wolfe1'),1);                             
        validator.setIntegerNonnegative("linesearch_nondescent_maxit")
        validator.setIntegerNonnegative("linesearch_reattempts")
        validator.setIntegerNonnegative("linesearch_reattempts_x0")
        validator.setRealInIntervalOO("linesearch_c_mu",0,1)   
        validator.setRealInIntervalOO("linesearch_c_mu_x0",0,1)    

        #  LOGGING PARAMETERS
        validator.setIntegerInRange("print_level",0,3)
        validator.setIntegerInRange("print_frequency",1,np.inf)
        validator.setIntegerInRange("print_width",9,23)
        validator.setLogical("print_ascii")
        validator.setLogical("print_use_orange")
                
        if hasattr(user_opts,"halt_log_fn") and user_opts.halt_log_fn != None:
            validator.setFunctionHandle("halt_log_fn")
        
        #  Extended ASCII chars in MATLAB on Windows are not monospaced so
        #  don't support them.
        opts = validator.getValidatedOpts()
        # if not opts.print_ascii:
        #     validator.assert(~ispc(),                                   ...
        #         'only opts.print_ascii == true is supported on Windows.');
        # end
    except Exception as e:
        print(e)


    #  GET THE VALIDATED OPTIONS AND POST PROCESS THEM
    opts = postProcess(n,validator.getValidatedOpts())
    
    #  For temperarily use
    # user_opts.__dict__.update(default_opts.__dict__)
    # opts = postProcess(n,user_opts)
    # dbg_print('pygransoOptions: currently assume all options are legal \n') 

    return opts

def postProcess(n,opts):
    
    # bump up the max fallback level if necessary
    if opts.max_fallback_level < opts.min_fallback_level:
        opts.max_fallback_level = opts.max_fallback_level
    
    
    # If an initial starting point was not provided, use random vector
    if np.all(opts.x0 == None):
        # opts.x0 = np.random.randn(n,1)
        rng = default_rng()
        opts.x0 = rng.standard_normal(size=(n,1))
    
    # If an initial inverse Hessian was not provided, use the identity
    if opts.H0 == None:
        opts.H0 = sparse.eye(n).toarray()
    
    
    if hasattr(opts,"QPsolver"):
        QPsolver = opts.QPsolver
    
    # % MATLAB default solver
    # skip MATLAB default solver and QPALM
    
    return opts


def getDefaults(n):
    [*_, LAST_FALLBACK_LEVEL] = pgC.pygransoConstants()

    # default options for GRANSO
    default_opts = Options()
    setattr(default_opts,'x0',None)
    # setattr(default_opts,'x0',np.ones((14,1)))  
    # dbg_print("pygransoOptions: set default x0 to a all ones for now. Original: None")
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
    dbg_print("pygransoOptions: set default maxit to a small # for now. Original: 1e-4. Fail for example 6. neighboorhood Cache line 81")
    setattr(default_opts,'evaldist',1e-4)
    # setattr(default_opts,'evaldist',1e-8)
    # dbg_print("pygransoOptions: set default maxit to a small # for now. Original: 1000")
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
    setattr(default_opts,'print_level',1)
    setattr(default_opts,'print_frequency',1)
    setattr(default_opts,'print_width',14)
    # Originally, false || ispc(). ispc returns logical 1 (true) if the version of MATLAB® software is for the Microsoft® Windows® platform.
    setattr(default_opts,'print_ascii',False)
    setattr(default_opts,'print_use_orange',True)
    setattr(default_opts,'halt_log_fn',None)
    setattr(default_opts,'debug_mode',False)

    return [default_opts, LAST_FALLBACK_LEVEL]
    