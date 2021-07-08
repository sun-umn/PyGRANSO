from private.makePenaltyFunction import PanaltyFuctions
from private import bfgsHessianInverse as bfgsHI, printMessageBox as pMB, pygransoPrinter as pP, bfgssqp, solveQP
from pygransoOptions import gransoOptions
import numpy as np
import copy

# only combined function allowed here. simpler form compare with GRANSO
# different cases needed if seperate obj eq and ineq are using
def processArguments(n,combined_fns,opts):
    problem_fns = combined_fns
    options = opts
    options = gransoOptions(n,options)
    return [problem_fns,options]

def getBfgsManager(opts):
    if opts.limited_mem_size == 0:
        get_bfgs_fn = lambda H,scaleH0, *_ : bfgsHI.bfgsHessianInverse(H,scaleH0)
        lbfgs_args  = None
        print("CAll BFGS: Skip LBFGS for now")
    else:
        print("LBFGS:TODO")
        # get_bfgs_fn = @bfgsHessianInverseLimitedMem;
        # lbfgs_args  = {     opts.limited_mem_fixed_scaling,     ...
        #                     opts.limited_mem_size,              ...
        #                     opts.limited_mem_warm_start         };
    


    
    bfgs_obj = get_bfgs_fn(opts.H0,opts.scaleH0,lbfgs_args)
    # DEBUG TEST:
    # print(bfgs_obj.getState() )
    # print(bfgs_obj.getCounts() )
    
    # remove potentially large and unnecessary data from the opts structure
    delattr(opts,'H0')
    delattr(opts,'limited_mem_warm_start')

    return [bfgs_obj,opts]

def printPrescalingMsg(prescaling_threshold,grad_norms,block_msg_fn):
    pass

def getPrescalingConstraintLines(type_str,c_large,width,cols):
    pass

def getTableRows(nums,num_width,cols,indent,brackets):
    pass

def quadprogInfoMsg():
    msg = ["PyGRANSO requires a quadratic program (QP) solver that has a quadprog-compatible ",
            "interface, as defined by Gurobi and QPALM..."]  
    return msg                   

def poorScalingDetectedMsgs():
    title = "POOR SCALING DETECTED"

    pre = ["The supplied problem appears to be poorly scaled at x0, which may adversely affect",
    "optimization quality.  In particular, the following functions have gradients whose",
    "norms evaluted at x0 are greater than 100:"] 

    post = ["NOTE: One may wish to consider whether the problem can be alternatively formulated",
    "with better inherent scaling, which may yield improved optimization results.",
    "Alternatively, GRANSO can optionally apply automatic pre-scaling to poorly-scaled",
    "objective and/or constraint functions if opts.prescaling_threshold is set to some",
    "sufficiently small positive number (e.g. 100).  For more details, see gransoOptions.",
    "",
    "To disable this notice, set opts.prescaling_info_msg = false."]
    return [title,pre,post]

def prescalingEnabledMsgs():
    title = "PRE-SCALING ENABLED"      

    pre = ["GRANSO has applied pre-scaling to functions whose norms were considered large at x0.",
        "GRANSO will now try to solve this pre-scaled version instead of the original problem",
        "given.  Specifically, the following functions have been automatically scaled",
        "downward so that the norms of their respective gradients evaluated at x0 no longer",
        "exceed opts.prescaling_threshold and instead are now equal to it:"] 

    post = ["NOTE: While automatic pre-scaling may help ameliorate issues stemming from when the",
        "objective/constraint functions are poorly scaled, a solution to the pre-scaled",
        "problem MAY OR MAY NOT BE A SOLUTION to the original unscaled problem.  One may wish",
        "to consider if the problem can be reformulated with better inherent scaling.  The",
        "amount of pre-scaling applied by GRANSO can be tuned, or disabled completely, via",
        "adjusting opts.prescaling_threshold.  For more details, see gransoOptions.",
        "To disable this notice, set opts.prescaling_info_msg = false."]  
    return [title,pre,post]

def getTerminationMsgLines(soln,constrained,width):
    pass

def getResultsLegend():
    s = "F = final iterate, B = Best (to tolerance), MF = Most Feasible"
    return s

def convergedToTolerancesMsg(constrained):
    if constrained:
        s = "converged to stationarity and feasibility tolerances."
    else:
        s = "converged to stationarity tolerance."
    return s

def progressSlowMsg(constrained):
    if constrained:
        s = "relative decrease in penalty function is below tolerance and feasibility tolerances satisfied."
    else:
        s = "relative decrease in objective function is below tolerance."
    return s

def targetValueAttainedMsg(constrained):
    if constrained:
        s = "target objective reached at point feasible to tolerances."
    else:
        s = "target objective reached."
    return s

def bracketedMinimizerFeasibleMsg(soln,constrained):
    if constrained:
        if soln.final.tv == 0:
            s2 = " at a (strictly) feasible point. "   
        else:
            s2 = " at a feasible point (to tolerances). "
    else:
        s2 = ". "
    
    s = [   "line search bracketed a minimizer but failed to satisfy Wolfe conditions",
            s2, "This may be an indication that approximate stationarity has been attained." ]
    return s

def bracketedMinimizerInfeasibleMsg():
    s = [  "line search bracketed a minimizer but failed to satisfy Wolfe conditions ",
            "at an infeasible point. The closest point encountered to the feasible ",
            "region is available in soln.most_feasible."]
    return s

def failedToBracketedMinimizerMsg(soln):
    if hasattr(soln,"mu_lowest"):
        s_mu = ["GRANSO attempted mu values down to {} unsuccessively.  However, if ".format(soln.mu_lowest),
                "the objective function is indeed bounded below on the feasible set, ",
                "consider restarting GRANSO with opts.mu0 set even lower than {}.".format(soln.mu_lowest)] 
    else:
        s_mu = ""

    s = ["line search failed to bracket a minimizer, indicating that the objective ",
        "function may be unbounded below. ",
        s_mu] 
    
    return s

def displayError(partial_computation,error_msg,err):
    pass

def displayErrorRecursive(err,full_report):
    pass

def partialComputationMsg():
    s = ["PyGRANSO: optimization halted on error.\n" ,
        "  - Output argument soln.final contains last accepted iterate prior to error\n",
        "  - See soln.error and console output below for more details on the cause of the error\n",
        "TODO..."]
    return s

def noComputationMsg():
    s = ["PyGRANSO: error on initialization.\n",
        "  - See console output below for more details on the cause of error.\n"]
    return s

def userSuppliedFunctionsErrorMsg():
    s = ["Please check your supplied routines defining the objective and constraints \n",
        "functions for correctness.\n"]
    return s

def quadprogErrorMsg():
    s = ["Incurring a quadprog error may be an indication of:\n",
        "  a) a weakness/flaw in the specific QP solver being used\n",
        "TODO" ]
    return s

def unknownErrorMsg():
    s = ["An unknown error has incurred.  ",
        "Please report it on PyGRANSO''s GitHub page.\n",
        " ",
        "    TODO"] 
    return s

def printSummaryAux(name,fieldname,soln,printer):
        if hasattr(soln,fieldname):
            printer.summary(name,getattr(soln,fieldname))

def pygranso(n,obj_fn,user_opts=None):
    """
    PyGRANSO: Python version GRadient-based Algorithm for Non-Smooth Optimization
    """
    
    # if nargin < 2
    #     error(  'GRANSO:inputArgumentsMissing',     ...
    #             'not all input arguments provided.' );
    # end
    
    # % First reset solveQP's persistent counters to zero
    # clear solveQP;

    
    # Initialization
    #  - process arguments
    #  - set initial Hessian inverse approximation
    #  - evaluate functions at x0

    try: 
        [problem_fns,opts] = processArguments(n,obj_fn,user_opts)
        # check realted function: np.matrix.H is recommened, consider np.transpose/conjugate 
        [bfgs_hess_inv_obj,opts] = getBfgsManager(opts)

        # construct the penalty function object and evaluate at x0
        # unconstrained problems will reset mu to one and mu will be fixed
        mPF = PanaltyFuctions() # make penalty functions 
        [ penaltyfn_obj, grad_norms_at_x0] =  mPF.makePenaltyFunction(opts, problem_fns)
    except Exception as e:
            print(e)   
            print("pygranso main loop Error")


    msg_box_fn = lambda margin_spaces,title_top,title_bottom,msg_lines,sides=True,user_width=0: pMB.printMessageBox(opts.print_ascii,opts.print_use_orange, margin_spaces,title_top,title_bottom,msg_lines,sides,user_width)

    print_notice_fn = lambda title,msg: msg_box_fn(2,title,"",msg,True) 
    if opts.print_level:
        print("\n")
        if opts.quadprog_info_msg:
            print_notice_fn('QP SOLVER NOTICE',quadprogInfoMsg())
        
        if opts.prescaling_info_msg:
            printPrescalingMsg( opts.prescaling_threshold,grad_norms_at_x0,print_notice_fn)
       
    printer = None
    if opts.print_level: 
        n_ineq          = penaltyfn_obj.numberOfInequalities()
        n_eq            = penaltyfn_obj.numberOfEqualities()
        constrained     = n_ineq or n_eq
        printer         = pP.pygransoPrinter(opts,n,n_ineq,n_eq)
    

    try:
        info = bfgssqp.bfgssqp(penaltyfn_obj,bfgs_hess_inv_obj,opts,printer)
    except Exception as e:
        print(e)   
        print("Error: pygranso bfgssqp ")
        # recover optimization computed so far
        penaltyfn_obj.restoreSnapShot()
    
    # package up solution in output argument
    [ soln, stat_value ]        = penaltyfn_obj.getBestSolutions()
    soln.H_final                = bfgs_hess_inv_obj.getState()
    soln.stat_value             = stat_value
    bfgs_counts                 = bfgs_hess_inv_obj.getCounts()
    soln.iters                  = bfgs_counts.requests
    soln.BFGS_updates           = bfgs_counts
    soln.fn_evals               = penaltyfn_obj.getNumberOfEvaluations()
    soln.termination_code       = info.termination_code
    [qp_requests,qp_errs]       = solveQP.solveQP('counts')
    qp_fail_rate                = 100 * (qp_errs / qp_requests)
    soln.quadprog_failure_rate  = qp_fail_rate
    if hasattr(info,"error"):
        soln.error              = info.error
    elif hasattr(info,"mu_lowest"):
        soln.mu_lowest          = info.mu_lowest

    # python version: new function for printSummary
    printSummary = lambda name,fieldname: printSummaryAux(name,fieldname,soln,printer)

    if opts.print_level:         
        printer.msg({ 'Optimization results:', getResultsLegend() })
        
        printSummary("F","final")
        printSummary("B","best")
        printSummary("MF","most_feasible")
        if penaltyfn_obj.isPrescaled():
            printer.unscaledMsg()
            printSummary("F","final_unscaled")
            printSummary("B","best_unscaled")
            printSummary("MF","most_feasible_unscaled")
        
        width = printer.msgWidth()
        printer.msg(getTerminationMsgLines(soln,constrained,width))

        if qp_fail_rate > 1:
            printer.quadprogFailureRate(qp_fail_rate)
        
        printer.close(); 
    
         
    if hasattr(soln,"error"):
        err = soln.error;      
        print("ERROR: In the end.")
    

    print("pygranso end")

    return -1

