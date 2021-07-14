from numpy.core.numeric import Inf
from private import pygransoConstants as pgC
from private.optionValidator import oV
import numpy as np
from scipy import sparse
from pygransoStruct import Options
from dbg_print import dbg_print
from private.isAnInteger import isAnInteger

def gransoOptions(n,options):
    """
    gransoOptions:
        Validate user options struct for granso.m.  If user_opts is [] or
        not provided, returned opts will be GRANSO's default parameters.
        Standard or advanced options may be set.  
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
        opts.x0 = np.random.randn(n,1)
    
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
    setattr(default_opts,'evaldist',1e-4)
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
    setattr(default_opts,'QPsolver',None)
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

    # setattr(default_opts,'QPsolver',None)

    return [default_opts, LAST_FALLBACK_LEVEL]
    