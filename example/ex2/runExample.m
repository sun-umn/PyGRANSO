function soln = runExample()
%   runExample: (examples/ex2)
%       Run GRANSO on a 2-variable nonsmooth Rosenbrock objective function,
%       subject to simple bound constraints.
%
%       Read this source code.
%
%       This tutorial example shows:
%
%           - how to set up anonymous functions to "preload" necessary
%             fixed data needed for the optimization functions, rather than
%             statically defining/loading such data directly in .m files
%
%           - how to set GRANSO's inputs when there aren't any equality
%             constraint functions (which also applies when there aren't
%             any inequality constraints)
%
%           - how to set some of GRANSO's parameters
%
%           - how to restart GRANSO from the last iterate of a previous
%             run (in either full-memory or limited-memory modes).
%
%   USAGE:
%       soln = runExample();
%
%   INPUT: [none]
%
%   OUTPUT:
%       soln        GRANSO's output struct
%
%   See also combinedFunction.
%


% key
var = {'x1','x2'};
% value: dimension. e.g., 2 by 2 => [2,2]
dim = {[1,1],[1,1]};
var_dim_map =  containers.Map(var, dim);

% calculate total number of scalar variables
nvar = 0;
for idx = 1:length(dim)
    curDim = dim(idx);
    nvar = nvar + curDim{1,1}(1)*curDim{1,1}(2);
end

% SET SOME GRANSO OPTIONS

% Set an initial point (which is infeasible)
opts.x0                 = [5.5; 5.5];

% By default GRANSO will print using extended ASCII characters to
% "draw" table borders.  However, MATLAB's diary function, which
% creates a log text file of the console output, does not retain these
% characters.  By setting opts.print_ascii to true, GRANSO will uses
% regular characters to print tables that can be captured by the diary
% command (albeit with not as nice looking output).
opts.print_ascii        = true;

% % By default, GRANSO prints an info message about QP solvers, since
% % GRANSO can be used with any QP solver that has a quadprog-compatible
% % interface.  Let's disable this message since we've already seen it
% % hundreds of times and can now recite it from memory.  ;-)
% opts.quadprog_info_msg  = false;

% Let's only do a short run of GRANSO, just 10 iterations initially.
opts.maxit              = 10;   % default is 1000

% GRANSO's penalty parameter is on the *objective* function, thus
% higher penalty parameter values favor objective minimization more
% highly than attaining feasibility.  Let's set GRANSO to start off
% with a higher initial value of the penalty parameter.  GRANSO will
% automatically tune the penalty parameter to promote progress towards
% feasibility.  GRANSO only adjusts the penalty parameter in a
% monotonically decreasing fashion.
opts.mu0                = 100;  % default is 1


% SET UP THE (ANONYMOUS) FUNCTION HANDLES

% First, we dynamically set the constant needed for the objective
parameters.w = 8;

% Embed the *current* value of w dynamically into the objective
% call mat2vec to enable GRANSO using matrix input
combined_fn = @(x) mat2vec(x,var_dim_map,nvar,parameters);
soln = granso(nvar,combined_fn);

fprintf('\nPress any key to restart GRANSO from the last iterate for up to 100 more iterations.\n');
pause


% NOW LET'S RESTART GRANSO FROM THE LAST ITERATE OF THE PREVIOUS RUN

% To restart GRANSO, we need to set the following three parameters:
%
%   1) opts.x0      - initial point
%   2) opts.mu0     - initial penalty parameter value
%   3) opts.H0      - initial value of the (approx) Hessian inverse
%
% to the their corresponding values for the last accepted iterate of
% the previous run.  (Note that if GRANSO has applied prescaling, then
% one must do additional processing to restart GRANSO from the exact
% same "prescaled" problem.  In other words, one must manually
% prescale the objective and constraint functions to match the
% prescaling that GRANSO applied.)

% Set the initial point and penalty parameter to their final values
% from the previous run
opts.x0 = soln.final.x;
opts.mu0 = soln.final.mu;


% PREPARE TO RESTART GRANSO IN FULL-MEMORY MODE
% Set the last BFGS inverse Hessian approximation as the initial
% Hessian for the next run.  Generally this is a good thing to do, and
% often it is necessary to retain this information when restarting (as
% on difficult nonsmooth problems, GRANSO may not be able to restart
% without it).  However, your mileage may vary.  In my testing, with
% the above settings, omitting H0 causes GRANSO to take an additional
% 16 iterations to converge on this problem.
opts.H0 = soln.H_final;     % try running with this commented out

% When restarting, soln.H_final may fail GRANSO's initial check to
% assess whether or not the user-provided H0 is positive definite.  If
% it fails this test, the test may be disabled by setting opts.checkH0
% to false.
%     opts.checkH0 = false;       % Not needed for this example

% If one desires to restart GRANSO as if it had never stopped (e.g.
% to continue optimization after it hit its maxit limit), then one must
% also disable scaling the initial BFGS inverse Hessian approximation
% on the very first iterate.
opts.scaleH0 = false;


% PREPARE TO RESTART GRANSO IN LIMITED-MEMORY MODE
% (Note that this example problem only has two variables!)
% If GRANSO was running in limited-memory mode, that is, if
% opts.limited_mem_size > 0, then GRANSO's restart procedure is
% slightly different, as soln.H_final will instead contain the most
% current L-BFGS state, not a full inverse Hessian approximation.
% Instead, do the following:
% 1) If you set a specific H0, you will need to set opts.H0 to whatever
%    you used previously.  By default, GRANSO uses the identity for H0.
% 2) Warm-start GRANSO with the most recent L-BFGS data by setting:
%    opts.limited_mem_warm_start = soln.H_final;
% NOTE: how to set opts.scaleH0 so that GRANSO will be restarted as if
% it had never terminated depends on the previously used values of
% opts.scaleH0 and opts.limited_mem_fixed_scaling.


% RESTART GRANSO
% Increase the maximum allowed iterations to 100
opts.maxit = 100;

% Restart GRANSO from the last iterate
soln = granso(nvar,combined_fn,opts);

end
