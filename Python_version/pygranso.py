from private import pygransoPrinter
from private.makePenaltyFunction import PanaltyFuctions
from private import bfgsHessianInverse as bfgsHI, printMessageBox as pMB
from private.bfgssqp import AlgBFGSSQP
from private.pygransoPrinter import pgP
from pygransoOptions import gransoOptions
from private.solveQP import getErr
import numpy as np
import copy
from dbg_print import dbg_print
from private.wrapToLines import wrapToLines
from time import sleep


def pygranso(n,obj_fn,user_opts=None):
    """
    PyGRANSO: Python version GRadient-based Algorithm for Non-Smooth Optimization

       Minimize a function, possibly subject to inequality and/or equality 
       constraints.  PyGRANSO is intended to be an efficient solver for 
       constrained nonsmooth optimization problems, without any special 
       structure or assumptions imposed on the objective or constraint 
       functions.  It can handle problems involving functions that are any
       or all of the following: smooth or nonsmooth, convex or nonconvex, 
       and locally Lipschitz or non-locally Lipschitz.  
 
       PyGRANSO only requires the objective and constraint functions. 

       The inequality constraints must be formulated as 'less than or
       equal to zero' constraints while the equality constraints must
       be formulated as 'equal to zero' constraints.  The user is 
       free to shift/scale these internally in order to specify how 
       hard/soft each individual constraints are, provided that they
       respectively remain 'less than or equal' or 'equal' to zero.
 
       The user must install a quadratic program solver,
       such as Gurobi and QPALM.  

       NOTE: 

       On initialization, PyGRANSO will throw errors if it detects invalid
       user options or if the user-provided functions to optimize either 
       do not evaluate or not conform to PyGRANSO's format.  However, once
       optimization begins, PyGRANSO will catch any error that is thrown and
       terminate normally, so that the results of optimization so far
       computed can be returned to the user.  This way, the error may be
       able to corrected by the user and PyGRANSO can be restarted from the
       last accepted iterate before the error occurred.
 
       After PyGRANSO executes, the user is expected to check all of the
       following fields:
           - soln.termination_code
           - soln.quadprog_failure_rate
           - soln.error (if it exists)
       to determine why PyGRANSO halted and to ensure that PyGRANSO ran
       without any issues that may have negatively affected its
       performance, such as QP solver failing too frequently or a
       user-provided function throwing an error or returning an invalid
       result.

       USAGE:
   
        - combined_fn evaluates objective and constraints simultaneously:
 
        "combined" format
       soln = pygranso(nvar,combined_fn)
       soln = pygranso(nvar,combined_fn,opts)
  
 
       INPUT:
       nvar               Number of variables to optimize.
 
        combinedFunction:
                      Function handle of single input X, a data structuture storing all input variables,
                       for evaluating:
 
                       - The values and gradients of the objective and
                         constraints simultaneously:
                         [f,ci,ce] = combinedFunction(X)
                         In this case, ci and/or ce should be returned as
                         None if no (in)equality constraints are given.

 
       options         Optional struct of settable parameters or None.
                       To see available parameters and their descriptions,
                       type:
                       >> help(pygransoOptions)
                       >> help(pygransoOptionsAdvanced)
 
       OUTPUT:
       soln            Structure containing the computed solution(s) and
                       additional information about the computation.
 
       If the problem has been pre-scaled, soln will contain:
 
       .scalings       Struct of pre-scaling multipliers of:
                         the objective          - soln.scalings.f
                         inequality constraints - soln.scalings.ci
                         equality constraints   - soln.scalings.ce
                       These subsubfields contain real-valued vectors of 
                       scalars in (0,1] that rescale the corresponding 
                       function(s) and their gradients.  A multiplier that
                       is one indicates that its corresponding function
                       has not been pre-scaled.  The absence of a 
                       subsubfield {f,ci,ce} indicates that none of the 
                       functions belonging to that group were pre-scaled.
 
       The soln struct returns the optimization results in the following
       fields:
 
       .final          Function values at the last accepted iterate
 
       .best           Info for the point that most optimizes the 
                       objective over the set of iterates encountered
                       during optimization.  Note that this point may not
                       be an accepted iterate.  For constrained problems,
                       this is the point that most optimizes the objective
                       over the set of points encountered that were deemed 
                       feasible with respect to the violation tolerances 
                       (opts.viol_ineq_tol and opts.viol_eq_tol).  If no 
                       points were deemed feasible to tolerances, then
                       this field WILL NOT BE present.
 
       .most_feasible  Info for the best most feasible iterate.  If
                       iterates are encountered which have zero total
                       violation, then this holds the best iterate amongst
                       those, i.e., the one which minimized the objective
                       function the most.  Otherwise, this contains the 
                       iterate which is closest to the feasible region, 
                       that is with the smallest total violation, 
                       regardless of the objective value.  NOTE: this 
                       field is only present in constrained problems.
 
       Note that if the problem was pre-scaled, the above fields will only
       contain the pre-scaled values.  Furthermore, the solution to the
       pre-scaled problem may or may not be a solution to the unscaled
       problem.  For convenience, corresponding unscaled versions of the
       above results are respectively provided in:
       .final_unscaled
       .best_unscaled
       .most_feasible_unscaled
 
       For .final, .best, and .most_feasible (un)scaled variants, the 
       following subsubfields are always present:
       .x                  the computed iterate
       .f                  value of the objective at this x
 
       If the problem is constrained, these subsubfields are also present:
       .ci                 inequality constraints at x
       .ce                 equality constraints at x
       .tvi                total violation of inequality constraints at x
       .tve                total violation of equality constraints at x
       .tv                 total violation at x (vi + ve)
       .feasible_to_tol    true if x is considered a feasible point with
                           respect to opts.viol_ineq_tol and
                           opts.viol_eq_tol
       .mu                 the value of the penalty parameter when this
                           this iterate was computed
 
       NOTE: The above total violation measures are with respect to the
       infinity norm, since the infinity norm is used for determining
       whether or not a point is feasible.  Note however that GRANSO 
       optimizes an L1 penalty function.
 
       The soln struct also has the following subfields:
       .H_final            Full-memory BFGS:
                           - The BFGS inverse Hessian approximation at
                             the final iterate
                           Limited-memory BFGS:
                           - A struct containing fields S,Y,rho,gamma, 
                             representing the final LBFGS state (so that
                             GRANSO can be warm started using this data).
       
       .stat_value         an approximate measure of stationarity at the 
                           last accepted iterate.  See opts.opt_tol, 
                           opts.ngrad, opts.evaldist, and equation (13) 
                           and its surrounding discussion in the paper 
                           referenced below describing the underlying 
                           BFGS-SQP method that GRANSO implements.
   
       .iters              The number of iterations incurred.  Note that 
                           if the last iteration fails to produce a step,
                           it will not be printed.  Thus, this value may 
                           sometimes appear to be greater by 1 compared to 
                           the last iterate printed.
 
       .BFGS_updates       A struct of data about the number of BFGS 
                           updates that were requested and accepted.
                           Numerical issues may force updates to be
                           (rarely) skipped.
       
       .fn_evals           The total number of function evaluations
                           incurred.  Evaluating the objective and
                           constraint functions at a single point counts
                           as one function evaluation.
   
       .termination_code   Numeric code indicating why GRANSO terminated:
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
               feasible set, consider restarting PyGRANSO with opts.mu0 set
               lower than soln.mu_lowest (see its description below for 
               more details).
 
           9:  PyGRANSO failed to produce a descent direction.
 
           10: NO LONGER IN USE (Previously, it was used to indicate if
               any of the user-supplied functions returned inf/NaN at x0.
               Now, PyGRANSO throws an error back to the user if this
               occurs).
 
           11: User-supplied functions threw an error which halted PyGRANSO.
 
           12: A quadprog error forced the steering procedure to be 
               aborted and PyGRANSO was halted (either because there were no 
               available fallbacks or opts.halt_on_quadprog_error was set 
               to true).  Only relevant for constrained problems.
 
           13: An unknown error occurred, forcing GRANSO to stop.  Please 
               report these errors to the developer.
 
        .quadprog_failure_rate  Percent of the time quadprog threw an error or
                           returned an invalid result.  From 0 to 100.
 
        .error                  Only present for soln.termination equal to 11, 
                           12, or 13.  Contains the thrown error that
                           caused GRANSO to halt optimization.
 
        .mu_lowest              Only present for constrained problems where
                           the line search failed to bracket a minimizer 
                           (soln.termination_code == 7).  This contains
                           the lowest value of mu tried in the line search
                           reattempts.  For more details, see the line
                           search user options by typing:
                           >> help(gransoOptionsAdvanced)
 
        CONSOLE OUTPUT:
       When opts.print_level is at least 1, PyGRANSO will print out
       the following information for each iteration:
   
       Iter            The current iteration number
 
       Penalty Function (only applicable if the problem is constrained)
           Mu          The current value of the penalty parameter 
           Value       Current value of the penalty function
           
           where the penalty function is defined as
               Mu*objective + total_violation
 
       Objective       Current value of the objective function
       
       Total Violation (only applicable if the problem is constrained)
           Ineq        Total violation of the inequality constraints
           Eq          Total violation of the equality constraints
       
           Both are l-infinity measures that are used to determine
           feasibility.
 
        Line Search
                         SD          Search direction type:
                           S   Steering with BFGS inverse Hessian
                           SI  Steering with identity 
                           QN  Standard BFGS direction 
                           GD  Gradient descent
                           RX  Random where X is the number of random 
                               search directions attempted 
           
                       Note that S and SI are only applicable for
                       constrained problems.
 
                       If the accepted search direction was obtained via a
                       fallback strategy instead of the standard strategy,
                       the search direction type will be printed in
                       orange.  The standard search direction types for
                       constrained and unconstrained problems respectively
                       is S and QN.  The standard search direction can be
                       modified via opts.min_fallback_level.  
 
                       The frequent use of fallbacks may indicate a
                       deficient or broken QP installation (or that 
                       the license is invalid or can't be verified).
           
           Evals       Number of points evaluated in line search
   
           t           Accepted length of line search step 
 
       Stationarity    Approximate stationarity measure
           Grads       Number of gradients used to compute measure
       
           Value       Value of the approximate stationarity measure.  
               
                       If fallbacks were employed to compute the
                       stationarity value, that is quadprog errors were 
                       encountered, its value will be printed in 
                       orange, with ':X' appearing after it.  The X 
                       indicates the number of requests to quadprog.  If
                       the value could not be computed at all, it will be
                       reported as Inf.
 
                       The frequent use of fallbacks may indicate a
                       deficient or broken quadprog installation (or that 
                       the license is invalid or can't be verified).
 
       See also gransoOptions, gransoOptionsAdvanced, makeHaltLogFunctions.
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
        pygransoPrinter_object = pgP()
        printer         = pygransoPrinter_object.pygransoPrinter(opts,n,n_ineq,n_eq)
    

    
    

    try:
        bfgssqp_obj = AlgBFGSSQP()
        info = bfgssqp_obj.bfgssqp(penaltyfn_obj,bfgs_hess_inv_obj,opts,printer)
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

    
    [qp_requests,qp_errs]       = getErr()
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
        
        sleep(0.1)
        
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
        
        printer.close()
    
         
    if hasattr(soln,"error"):
        err = soln.error;      
        print("ERROR: In the end.")
    

    # print("pygranso end")

    return soln




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
        dbg_print("CAll BFGS: Skip LBFGS for now")
    else:
        dbg_print("LBFGS:TODO")
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
    if soln.termination_code == 0:
        s = convergedToTolerancesMsg(constrained)
    elif soln.termination_code ==  1:
        s = progressSlowMsg(constrained)
    elif soln.termination_code ==  2:
        s = targetValueAttainedMsg(constrained)
    elif soln.termination_code ==  3:
        s = "halt signaled by user via opts.halt_log_fn."
    elif soln.termination_code ==  4:
        s = "max iterations reached."
    elif soln.termination_code ==  5:
        s = "clock/wall time limit reached."
    elif soln.termination_code ==  6:
        s = bracketedMinimizerFeasibleMsg(soln,constrained);          
    elif soln.termination_code ==  7:
        s = bracketedMinimizerInfeasibleMsg()
    elif soln.termination_code ==  8:
        s = failedToBracketedMinimizerMsg(soln)
    elif soln.termination_code ==  9:
        s = "failed to produce a descent direction."
    #  Case 10 is no longer used
    elif soln.termination_code ==  11:
        s = "user-supplied functions threw an error."
    elif soln.termination_code ==  12:
        s = "steering aborted due to quadprog() error; see opts.halt_on_quadprog_error."
    else:
        s = "unknown termination condition."
    
    s = "PyGRANSO termination code: %d --- %s" % (soln.termination_code,s)    
    lines = [   "Iterations:              %d" % (soln.iters),
                "Function evaluations:    %d" % (soln.fn_evals)]
    lines.extend( wrapToLines(s,width,0))

    return lines

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

