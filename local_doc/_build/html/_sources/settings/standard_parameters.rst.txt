Standard Parameters
========================

x0
----------------
n by 1 double precision torch tensor. Default value: torch.randn(n,1).to(device=torch_device, dtype=torch.double)

Initial starting point. One should pick x0 such that the objective
and constraint functions are smooth at and about x0. If this is
difficult to ascertain, it is generally recommended to initialize
PyGRANSO at randomly-generated starting points.

mu0
----------------
Positive real value. Default value: 1

Initial value of the penalty parameter. 
NOTE: irrelevant for unconstrained optimization problems.

H0
----------------
n by n double precision torch tensor. Default value: torch.eye(n,device=torch_device, dtype=torch.double) 

Initial inverse Hessian approximation.  In full-memory mode, and 
if opts.checkH0 is true, PyGRANSO will numerically assert that this
matrix is positive definite. In limited-memory mode, that is, if
opts.limited_mem_size > 0, no numerical checks are done but this 
matrix must be a sparse matrix.

checkH0
----------------
Boolean value. Default value: True

By default, PyGRANSO will check whether or not H0 is numerically
positive definite (by checking whether or not cholesky() succeeds).
However, when restarting PyGRANSO from the last iterate of an earlier
run, using soln.H_final (the last BFGS approximation to the inverse
Hessian), soln.H_final may sometimes fail this check.  Set this
option to False to disable it. No positive definite check is done
when limited-memory mode is enabled.

scaleH0
----------------
Boolean value. Default value: True

Scale H0 during BFGS/L-BFGS updates.  For full-memory BFGS, scaling
is only applied on the first iteration only, and is generally only
recommended when H0 is the identity (which is PyGRANSO's default).
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

By default, PyGRANSO uses full-memory BFGS updating.  For nonsmooth
problems, full-memory BFGS is generally recommended.  However, if
this is not feasible, one may optionally enable limited-memory BFGS
updating by setting opts.limited_mem_size to a positive integer
(significantly) less than the number of variables.

limited_mem_fixed_scaling
--------------------------------
Boolean value. Default value: True

In contrast to full-memory BFGS updating, limited-memory BFGS
permits that H0 can be scaled on every iteration.  By default,
PyGRANSO will reuse the scaling parameter that is calculated on the
very first iteration for all subsequent iterations as well.  Set
this option to False to force PyGRANSO to calculate a new scaling
parameter on every iteration.  Note that opts.scaleH0 has no effect
when opts.limited_mem_fixed_scaling is set to True.

limited_mem_warm_start
--------------------------------
Python dictionary with key to be 'S', 'Y', 'rho' and 'gamma'. Default value: None
       
If one is restarting PyGRANSO, the previous L-BFGS information can be
recycled by setting opts.limited_mem_warm_start = soln.H_final,
where soln is PyGRANSO's output struct from a previous run.  Note
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

Prints a notice that PyGRANSO has either automatically pre-scaled at
least one of the objective or constraint functions or it has
deteced that the optimization problem may be poorly scaled.  For
more details, see opts.prescaling_threshold.  

opt_tol     
----------------        
Positive real value. Default value: 1e-8

Tolerance for reaching (approximate) optimality/stationarity.
See opts.ngrad, opts.evaldist, and the description of PyGRANSO's 
output argument soln, specifically the subsubfield .dnorm for more
information.

rel_tol
----------------
Non-negative real value. Default value: 0

Tolerance for determining when the relative decrease in the penalty
function is sufficiently small.  PyGRANSO will terminate if when 
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
PyGRANSO's measure of stationarity requires a history of previous 
gradients.  Note that large values of ngrad can make the related QP
expensive to solve, if a significant fraction of the currently
cached gradients were evaluated at points within evaldist of the 
current iterate.  Using 1 is recommended if and only if the problem 
is unconstrained and the objective is known to be smooth.  See 
opts.opt_tol, opts.evaldist, and the description of PyGRANSO's output
argument soln, specifically the subsubfield .dnorm for more
information.

evaldist
----------------                       
Positive real value. Default value: 1e-4

Previously evaluated gradients are only used in the stationarity 
test if they were evaluated at points that are within distance 
evaldist of the current iterate x.  See opts.opt_tol, opts.ngrad, 
and the description of PyGRANSO's output argument soln, specifically 
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
----------------          
Boolean value. Default value: True

If the line search brackets a minimizer but fails to satisfy the 
weak Wolfe conditions (necessary for a step to be accepted), PyGRANSO 
will terminate at this iterate when this option is set to true 
(default).  For unconstrained nonsmooth problems, it has been 
observed that this type of line search failure is often an 
indication that a stationarity has in fact been reached.  By 
setting this parameter to False, PyGRANSO will instead first attempt 
alternative optimization strategies (if available) to see if
further progress can be made before terminating.   See
gransoOptionsAdvanced for more details on PyGRANSO's available 
fallback optimization strategies and how they can be configured. 

quadprog_info_msg
--------------------------------
Boolean value. Default value: True

Prints a notice that PyGRANSO's requires a quadprog-compatible QP
solver and that the choice of QP solver may affect PyGRANSO's quality
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

By default, PyGRANSO's printed output uses the extended character map, 
so nice looking tables can be made.  But if you need to record the output, 
you can restrict the printed output to only use the basic ASCII character map


print_use_orange   
--------------------------------
Boolean value. Default value: True

PyGRANSO's uses orange
printing to highlight pertinent information.  However, the user
is the given option to disable it if they need to record the output

halt_log_fn
--------------------------------
Lambda Function. Default value: None

A user-provided function handle that is called on every iteration
to allow the user to signal to PyGRANSO for it to halt at that 
iteration and/or create historical logs of the progress of the
algorithm. 





