from numpy import double
import torch
from pygranso.private.makePenaltyFunction import PanaltyFuctions
from pygranso.private import bfgsHessianInverse as bfgsHI, printMessageBox as pMB, bfgsHessianInverseLimitedMem as lbfgsHI
from pygranso.private.bfgssqp import AlgBFGSSQP
from pygranso.private.pygransoPrinter import pgP
from pygranso.pygransoOptions import pygransoOptions
from pygranso.private.solveQP import getErr
from pygranso.private.wrapToLines import wrapToLines
from time import sleep
from pygranso.private.tensor2vec import tensor2vec
from pygranso.private.getNvar import getNvar, getNvarTorch
from pygranso.private.processVarSpec import processVarSpec
import traceback,sys

def pygranso(var_spec,combined_fn,user_opts=None):
    """
    PyGRANSO: A PyTorch-enabled port of GRANSO with auto-differentiation

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

        The user must install a quadratic program solver such as OSQP.

        PyGRANSO uses modifed versions of the BFGS inverse Hessian approximation
        update formulas and the inexact weak Wolfe line search from HANSO v2.1.
        See the documentation of HANSO for more information on the use of
        quasi-Newton methods for nonsmooth unconstrained optimization.

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
            soln = pygranso(var_spec,combined_fn,user_opts=None)

            NOTE: PyGRANSO differs from GRANSO in that pygranso only supports
                  combined_fn for specifying the objective and constraints
                  functions, as opposed to the alternative obj_fn, ineq_fn,
                  eq_fn function handles that GRANSO can also use.

        INPUT:

        var_spec
            var_spec can be one of the following two values:

            1. var_dim_map: used in general cases

            OR

            2.[var_dim_map,nn_model]: if torch neural network model used in both objective and constraints,
            we need additional nn_model arg to allow autodifferentiation.

            var_dim_map:
                            Required

                            A dictionary for optimization variable information,
                            where the key is the variable name and val is a list for correpsonding dimension:
                            e.g., var_in = {"x": [3]}; var_in = {"U": [5,10], "V": [10,20]; var_in = {"W":[3,28,28]}}

                            It should not be used when nn_model is specfied, as optimization variable information can be
                            obtained from neural network model

            nn_model:
                            Default: None

                            Neural network model defined by torch.nn. It only used when torch.nn was used to
                            define the combined_fn

        combined_fn:
                Function handle of single input X, a data structuture storing all input variables,
                for evaluating:

                - The values of the objective and constraints simultaneously:
                    [f,ci,ce] = combined_fn(X)

                    ci and/or ce should be returned as
                    None if no (in)equality constraints are given.

                    Auto-differentiation is used to obtain gradients automatically.

                OR (for advanced user who want to provide explicit gradients or
                    use AD for some gradients but not all)

                - The values of the objective, constraints, and their gradients simultaneously:
                    [f,f_grad,ci,ci_grad,ce,ce_grad] = combined_fn(X)

                    ci and/or ce (and their corresponding gradients) should be returned as
                    None if no (in)equality constraints are given.

                    In this case, the requirements on [f,f_grad,ci,ci_grad,ce,ce_grad]
                    are the same as GRANSO:

                    Each function handle returns the value of the function(s)
                    evaluated at single input X, a data structuture storing all input variables,
                    along with its corresponding gradient(s) as a matrix of column vectors.
                    For example, if there are n variables and p inequality
                    constraints, then ci must be supplied as a column vector in
                    R^p while ci_grad must be given as an n by p matrix of p
                    gradients for the p inequality constraints.

                    NOTE: This explicit gradient interface may change in a future release of PyGRANSO
                          to be more consistent with the AD interface.

        user_opts:
                        Optional struct of settable parameters or None.
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
        whether or not a point is feasible.  Note however that PyGRANSO
        optimizes an L1 penalty function.

        The soln struct also has the following subfields:
        .H_final            Full-memory BFGS:
                            - The BFGS inverse Hessian approximation at
                                the final iterate
                            Limited-memory BFGS:
                            - A struct containing fields S,Y,rho,gamma,
                                representing the final LBFGS state (so that
                                PyGRANSO can be warm started using this data).

        .stat_value         an approximate measure of stationarity at the
                            last accepted iterate.  See opts.opt_tol,
                            opts.ngrad, opts.evaldist, and equation (13)
                            and its surrounding discussion in the paper
                            referenced below describing the underlying
                            BFGS-SQP method that PyGRANSO implements.

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

        .termination_code   Numeric code indicating why PyGRANSO terminated:
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

            13: An unknown error occurred, forcing PyGRANSO to stop.  Please
                report these errors to the developer.

        .quadprog_failure_rate  Percent of the time quadprog threw an error or
                            returned an invalid result.  From 0 to 100.

        .error                  Only present for soln.termination equal to 11,
                            12, or 13.  Contains the thrown error that
                            caused PyGRANSO to halt optimization.

        .mu_lowest              Only present for constrained problems where
                            the line search failed to bracket a minimizer
                            (soln.termination_code == 7).  This contains
                            the lowest value of mu tried in the line search
                            reattempts.  For more details, see the line
                            search user options by typing:
                            >> help(pygransoOptionsAdvanced)

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

        See also pygransoOptions, pygransoOptionsAdvanced.

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

        NOTE: PyGRANSO in all capitals refers to the software package and is the
        form that should generally be used.  pygranso or pygranso.py in lowercase
        letters refers specifically to the PyGRANSO routine/command.

        pygranso.py (introduced in PyGRANSO v1.0.0)
        Copyright (C) 2016-2021 Tim Mitchell and Buyun Liang

        This file is a MATLAB-to-Python port of granso.m from
        GRANSO v1.6.4 with the following new functionality and/or changes:
            1. Adding new options to handle pytorch neural network model.
            2. Adding f_eval_fn to allow cheaper backtracking line search, as
               eval gradient is not needed in backtracking line search.
            3. Add torch_device and double precision options to allow user
               select cuda/cpu and double/float.
        Ported from MATLAB to Python and modified by Buyun Liang, 2021

        For comments/bug reports, please visit the PyGRANSO webpage:
        https://github.com/sun-umn/PyGRANSO

        PyGRANSO Version 1.0.0, 2021, see AGPL license info below.

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

    # Initialization
    #  - process arguments
    #  - set initial Hessian inverse approximation
    #  - evaluate functions at x0

    [var_dim_map,nn_model] = processVarSpec(var_spec)
    try:
        if nn_model != None:
            n = getNvarTorch(nn_model.parameters())
            
        else:
            # call the functions getNvar to get the total number of (scalar) variables
            n = getNvar(var_dim_map)

        opts = pygransoOptions(n,user_opts)
        torch_device = opts.torch_device

        if nn_model != None:
            problem_fns = lambda x: tensor2vec(combined_fn ,x,var_dim_map,n,torch_device, model = nn_model,double_precision=opts.double_precision,globalAD=opts.globalAD)
            if opts.is_backtrack_linesearch == True:
                f_eval_fn = lambda x: tensor2vec(combined_fn ,x,var_dim_map,n,torch_device, model = nn_model,double_precision=opts.double_precision,get_grad=False,globalAD=opts.globalAD)
            else:
                f_eval_fn = None

        else:
            n = getNvar(var_dim_map)
            problem_fns = lambda x: tensor2vec(combined_fn ,x,var_dim_map,n,torch_device, double_precision=opts.double_precision,globalAD=opts.globalAD)
            if opts.is_backtrack_linesearch == True:
                f_eval_fn = lambda x: tensor2vec(combined_fn ,x,var_dim_map,n,torch_device, model = nn_model,double_precision=opts.double_precision,get_grad=False,globalAD=opts.globalAD)
            else:
                f_eval_fn = None

        [bfgs_hess_inv_obj,opts] = getBfgsManager(opts,torch_device,opts.double_precision)
        # construct the penalty function object and evaluate at x0
        # unconstrained problems will reset mu to one and mu will be fixed
        mPF = PanaltyFuctions() # make penalty functions
        [penaltyfn_obj,grad_norms_at_x0] =  mPF.makePenaltyFunction(opts, f_eval_fn, problem_fns, torch_device = torch_device, double_precision=opts.double_precision)
    except Exception as e:
        print(traceback.format_exc())
        sys.exit()

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
        info = bfgssqp_obj.bfgssqp(penaltyfn_obj,bfgs_hess_inv_obj,opts,printer, torch_device)
    except Exception as e:
        print(traceback.format_exc())
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
    if qp_requests == 0:
        qp_fail_rate = 0
    else:
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
        sleep(0.0001) # Prevent race condition
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
        err = soln.error
        print("ERROR: In the end of main loop.")
        print(err)


    return soln

# # only combined function allowed here. simpler form compare with pygranso
# # different cases needed if seperate obj eq and ineq are using
# def processArguments(n,combined_fns,opts,torch_device):
#     problem_fns = combined_fns
#     options = opts
#     options = pygransoOptions(n,options,torch_device)
#     return [problem_fns,options]

def getBfgsManager(opts,torch_device,double_precision):
    if opts.limited_mem_size == 0:
        get_bfgs_fn = lambda H,scaleH0 : bfgsHI.bfgsHessianInverse(H,scaleH0)
    else:
        get_bfgs_fn = lambda H,scaleH0 : lbfgsHI.bfgsHessianInverseLimitedMem(H,scaleH0,opts.limited_mem_fixed_scaling,opts.limited_mem_size,opts.limited_mem_warm_start,torch_device,double_precision)

    bfgs_obj = get_bfgs_fn(opts.H0,opts.scaleH0)

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
    msg = ["PyGRANSO requires a quadratic program (QP) solver that has a quadprog-compatible interface,",
            "the default is osqp. Users may provide their own wrapper for the QP solver.", ""
            "To disable this notice, set opts.quadprog_info_msg = False"]
    return msg

def poorScalingDetectedMsgs():
    title = "POOR SCALING DETECTED"

    pre = ["The supplied problem appears to be poorly scaled at x0, which may adversely affect",
    "optimization quality.  In particular, the following functions have gradients whose",
    "norms evaluted at x0 are greater than 100:"]

    post = ["NOTE: One may wish to consider whether the problem can be alternatively formulated",
    "with better inherent scaling, which may yield improved optimization results.",
    "Alternatively, PyGRANSO can optionally apply automatic pre-scaling to poorly-scaled",
    "objective and/or constraint functions if opts.prescaling_threshold is set to some",
    "sufficiently small positive number (e.g. 100).  For more details, see pygransoOptions.",
    "",
    "To disable this notice, set opts.prescaling_info_msg = false."]
    return [title,pre,post]

def prescalingEnabledMsgs():
    title = "PRE-SCALING ENABLED"

    pre = ["PyGRANSO has applied pre-scaling to functions whose norms were considered large at x0.",
        "PyGRANSO will now try to solve this pre-scaled version instead of the original problem",
        "given.  Specifically, the following functions have been automatically scaled",
        "downward so that the norms of their respective gradients evaluated at x0 no longer",
        "exceed opts.prescaling_threshold and instead are now equal to it:"]

    post = ["NOTE: While automatic pre-scaling may help ameliorate issues stemming from when the",
        "objective/constraint functions are poorly scaled, a solution to the pre-scaled",
        "problem MAY OR MAY NOT BE A SOLUTION to the original unscaled problem.  One may wish",
        "to consider if the problem can be reformulated with better inherent scaling.  The",
        "amount of pre-scaling applied by PyGRANSO can be tuned, or disabled completely, via",
        "adjusting opts.prescaling_threshold.  For more details, see pygransoOptions.",
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

    s = "PyGRANSO termination code: %d --- %s" % (soln.termination_code,"".join(s))
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
        s_mu = ["PyGRANSO attempted mu values down to {} unsuccessively.  However, if ".format(soln.mu_lowest),
                "the objective function is indeed bounded below on the feasible set, ",
                "consider restarting PyGRANSO with opts.mu0 set even lower than {}.".format(soln.mu_lowest)]
    else:
        s_mu = ""

    s = ["line search failed to bracket a minimizer, indicating that the objective ",
        "function may be unbounded below. "] + s_mu

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
