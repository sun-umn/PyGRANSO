from pygransoStruct import genral_struct
from private import pygransoConstants as pC, gransoPrinterColumns as gPC

def prescalingEndBlockMsg():
    s = ["",
        "PyGRANSO applied pre-scaling at x0.  Information:",
        " - ABOVE shows values for the pre-scaled problem",
        " - BELOW shows the unscaled values for the optimization results.",
        "NOTE: the pre-scaled solution MAY NOT be a solution to the original",
        "unscaled problem!  For more details, see opts.prescaling_threshold.",
        ""]

    return s

def quadprogFailureRateMsg(rate):
    s = ["WARNING: PyGRANSO''s performance may have been hindered by issues with QP solver.",
    "quadprog''s failure rate: {}%".format(rate),
    "Ensure that quadprog is working correctly!"]
    return s

def pygransoPrinter(opts,n,n_ineq,n_eq):
    """    
    pygransoPrinter:
    Object for handling printing out info for each iteration and
    messages.
    """
    # Setup printer options from GRANSO options
    ascii                           = opts.print_ascii
    use_orange                      = opts.print_use_orange
    print_opts = genral_struct()
    setattr(print_opts,"use_orange",use_orange)
    setattr(print_opts,"print_width",opts.print_width)
    setattr(print_opts,"maxit",opts.maxit)
    setattr(print_opts,"ls_max_estimate",50*(opts.linesearch_reattempts + 1))

    [*_,LAST_FALLBACK_LEVEL]         = pC.pygransoConstants()
    if opts.max_fallback_level < LAST_FALLBACK_LEVEL:
        setattr(print_opts,"random_attempts",0)
    else:
        setattr(print_opts,"random_attempts",opts.max_random_attempts)
    print_opts.ngrad                = opts.ngrad + 1

    cols        = gPC.pygransoPrinterColumns(print_opts,n_ineq, n_eq)
    constrained = n_ineq || n_eq;
    if constrained
        pen_label           = 'Penalty Function';
        viol_label          = 'Total Violation';
        pen_vals_fn         = @penaltyFunctionValuesConstrained;
        if n_ineq && n_eq
            viol_vals_fn    = @violationValuesBoth;
        elseif n_ineq
            viol_vals_fn    = @violationValuesInequality;
        else
            viol_vals_fn    = @violationValuesEquality;
        end 
    else
        pen_label           = 'Penalty Fn';
        viol_label          = 'Violation';
        pen_vals_fn         = @penaltyFunctionValues;
        viol_vals_fn        = @violationValues;
    end

    printer = -1
    print("gransoPrinter TODO!")
    return printer