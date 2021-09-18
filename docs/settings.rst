Settings
========

Below listed the key options. For more options, please check the documentation of pygransoOptions.py and 
pygransoOptionsAdvanced.py in the source code.

x0
----------------

n by 1 real numpy array. Default value: rng.standard_normal(size=(n,1))

Initial starting point.  One should pick x0 such that the objective
and constraint functions are smooth at and about x0.  If this is
difficult to ascertain, it is generally recommended to initialize
PyGRANSO at randomly-generated starting points.

mu0
----------------
Positive real value. Default value: 1

Initial value of the penalty parameter. 
NOTE: irrelevant for unconstrained optimization problems.


H0
----------------

n by n real numpy array. Default value: scipy.sparse.eye(n).toarray()

Initial inverse Hessian approximation.  In full-memory mode, and 
if opts.checkH0 is true, PyGRANSO will numerically assert that this
matrix is positive definite.

checkH0
----------------

Boolean value. Default value: True

By default, PyGRANSO will check whether or not H0 is numerically
positive definite (by checking whether or not chol() succeeds).
However, when restarting PyGRANSO from the last iterate of an earlier
run, using soln.H_final (the last BFGS approximation to the inverse
Hessian), soln.H_final may sometimes fail this check.  Set this
option to False to disable it.

opt_tol     
----------------        

Positive real value. Default value: 1e-8

Tolerance for reaching (approximate) optimality/stationarity.
See opts.ngrad, opts.evaldist, and the description of PyGRANSO's 
output argument soln, specifically the subsubfield .dnorm for more
information.

fvalquit
----------------
Positive real value. Default value: -Inf

Quit if objective function drops below this value at a feasible 
iterate (that is, satisfying feasibility tolerances 
opts.viol_ineq_tol and opts.viol_eq_tol).

print_level     
----------------
Integer in {0,1,2,3}. Default value: 1

Level of detail printed to console regarding optimization progress:

0 - no printing whatsoever

1 - prints info for each iteration  

2 - adds additional info about BFGS updates and line searches (TODO)

3 - adds info on any errors that are encountered (TODO)

print_frequency      
----------------          

Positive integer. Default value: 1

Sets how often the iterations are printed.  When set to 1, every
iteration is printed; when set to 10, only every 10th iteration is
printed.  When set to Inf, no iterations are printed, except for
at x0.  Note that this only affects .print_level == 1 printing;
all messages from higher values of .print_level will still be
printed no matter what iteration they occurred on.

print_print_ascii     
----------------          

Boolean value. Default value: False

By default, PyGRANSO's printed output uses the extended character map, 
so nice looking tables can be made.  But if you need to record the output, 
you can restrict the printed output 
to only use the basic ASCII character map

maxit
----------------

Positive integer. Default value: 1000

Max number of iterations.

halt_on_linesearch_bracket     
----------------          

Boolean value. Default value: True

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
fallback optimization strategies and how they can be configured. Recommend setting False in deep learning problem.

max_fallback_level     
----------------        

Positive integer. Default value: 4

Max number of strategy to be employed (>= min_fallback_level)
NOTE: fallback levels 0 and 1 are only relevant for constrained problems. 

        SEARCH DIRECTION STRATEGIES
        If a step cannot be taken with the current search direction (e.g.
        computed an invalid search direction or the line search failed on a
        valid search direction), PyGRANSO may attempt up to four optional 
        fallback strategies to try to continue making progress from the current
        iterate.  The strategies are as follows and are attempted in order:
            
            0. BFGS-SQP steering 
                - default for constrained problems
                - irrelevant for unconstrained problems
            1. BFGS-SQP steering with BFGS's inverse Hessian approximation
                replaced by the identity. If strategy #0 failed because quadprog
                failed on the QPs, this "steepest descent" version of the 
                steering QPs may be easier to solve.
                - irrelevant for unconstrained problems
            2. Standard BFGS update on penalty/objective function, no steering
                - default for unconstrained problems
            3. Steepest descent on penalty/objective function, no steering
            4. Randomly generated search direction 

QPsolver
------------------

String in {'osqp', 'gurobi'}. Default value: 'osqp'

Select the QP solver used in the steering strategy and termination condition.

init_step_size     
----------------        

Positive real value. Default value: 1

Initial step size t in line search method. Recommend using small value (e.g., 1e-2) for deep learning problem.

init_step_size     
----------------        

Positive integer. Default value: inf

Max number of iterations in line search method. Recommend using small value (e.g., 25) for deep learning problem.

is_backtrack_linesearch     
----------------          

Boolean value. Default value: False

By default, PyGRANSO will use Weak-Wolfe line search method. By enabling this method, the second wolfe condition will be disabled.

searching_direction_rescaling     
----------------          

Boolean value. Default value: False

Rescale the norm of searching direction to be 1. Recommend setting True in deep learning problem.

disable_terminationcode_6     
----------------          

Boolean value. Default value: False

Disable termination code 6 to ensure pygranso can always make a movement even if the line search failed. Recommend setting True in deep learning problem.




