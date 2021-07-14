from numpy.core.numeric import Inf
from private import pygransoConstants as pgC
from private import optionValidator as oV
import numpy as np
from scipy import sparse
from pygransoStruct import Options
from dbg_print import dbg_print

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
    validator = oV.optionValidator('PyGRANSO',default_opts)
    # validator.setUserOpts(user_opts);

    # surround the validation so we can rethrow the error from GRANSO
    # try ...
    # % GET THE VALIDATED OPTIONS AND POST PROCESS THEM
    # opts = postProcess(validator.getValidatedOpts());
    
    user_opts.__dict__.update(default_opts.__dict__)
    #  Tobedel later
    opts = postProcess(n,user_opts)

    dbg_print('pygransoOptions: currently assume all options are legal \n') 
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
    
    
    # if isfield(opts.quadprog_opts,'QPsolver')
    #     QPsolver = opts.quadprog_opts.QPsolver;
    # end
    
    # % MATLAB default solver
    # if (strcmp(QPsolver,'quadprog'))
        
    #     % By default, suppress quadprog's console printing and warnings
    #     if ~isfield(opts.quadprog_opts,'Display')
    #         opts.quadprog_opts.Display = 'off';
    #     end
        
    #     % Technically the following is a solveQP option, not a quadprog one
    #     if ~isfield(opts.quadprog_opts,'suppress_warnings')
    #         opts.quadprog_opts.suppress_warnings = true;
    #     end
        
    # %         Update: By default, suppress qpalm's console printing and warnings
    # elseif (strcmp(QPsolver,'qpalm'))
    #     if ~isfield(opts.quadprog_opts,'print_iter')
    #         opts.quadprog_opts.verbose = false;
    #     end
    # end
    
    return opts


def getDefaults(n):
    [*_, LAST_FALLBACK_LEVEL] = pgC.pygransoConstants()

    # default options for GRANSO
    default_opts = Options()
    # setattr(default_opts,'x0',None)
    setattr(default_opts,'x0',np.ones((14,1)))  
    dbg_print("pygransoOptions: set default x0 to a all ones for now. Original: None")

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

    dbg_print("pygransoOptions: set default maxit to a small # for now. Original: 1000")
    
    setattr(default_opts,'maxit',30)
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
    setattr(default_opts,'quadprog_opts',None)
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
    