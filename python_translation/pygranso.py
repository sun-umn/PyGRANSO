from private import makePenaltyFunction as mPF, bfgsHessianInverse as bfgsHI
from pygransoOptions import gransoOptions
import numpy as np

def pygranso(n,combined_fns,opts=None):
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
        [problem_fns,opts] = processArguments(n,combined_fns,opts)
        [bfgs_hess_inv_obj,opts] = getBfgsManager(opts)
      
        # construct the penalty function object and evaluate at x0
        # unconstrained problems will reset mu to one and mu will be fixed
        [ penaltyfn_obj, grad_norms_at_x0] =  mPF.makePenaltyFunction(opts, problem_fns)
    except ValueError:
    #         catch err
    #     switch err.identifier
    #         case 'GRANSO:invalidUserOption'
    #             printRed('GRANSO: invalid user option.\n');
    #             err.throwAsCaller();
    #         case 'GRANSO:userSuppliedFunctionsError'
    #             displayError(false,userSuppliedFunctionsErrorMsg(),err);
    #             err.cause{1}.rethrow();
    #         otherwise
    #             printRed('GRANSO: ');
    #             printRed(unknownErrorMsg());
    #             fprintf('\n');
    #             err.rethrow();
    #     end
    # end
        print("err")


    print("pygranso end")

    return -1

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