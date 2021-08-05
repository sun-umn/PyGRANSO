function [soln,log] = runExample()
%   runExample: (examples/ex3)
%       Run GRANSO on a 2-variable nonsmooth Rosenbrock objective function,
%       subject to simple bound constraints.
%    
%       Read this source code.
%  
%       This tutorial example shows:
%
%           - how to use GRANSO's "combined" format to easily share 
%             expensive computed results that are needed in both the
%             objective and inequality constraints
% 
%           - how to use halt_log_template/makeHaltLogFunctions.m and
%             opts.halt_log_fn to create a history of iterates
%
%           - the importance of using an initial point that is neither near
%             nor on a nonsmooth manifold, that is, the functions 
%             (objective and constraints) should be smooth at and *about* 
%             the initial point.
%
%       NOTE: In practice, it may be hard/expensive to check whether or not
%       an initial point is on or near a nonsmooth manifold.  As a
%       consequence, in lack of more information, we often recommend
%       initializing GRANSO from randomly generated or at least randomly
%       perturbed starting points.
%
%   USAGE:
%       soln = runExample();
% 
%   INPUT: [none]
%   
%   OUTPUT:
%       soln        GRANSO's output struct
%   
%       log         Struct containing a history of the iterates with the
%                   following fields:
%
%       .x          cell array of the accepted points
%
%       .f          vector of the corresponding objective values
%
%       .v          vector of the corresponding violation values, for
%                   testing feasibility (infinity norm).
% 
%   See also combinedFunction, makeHaltLogFunctions.
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

% Set a randomly generated starting point.  In theory, with probability
% one, a randomly selected point will not be on a nonsmooth manifold.
opts.x0     = randn(nvar,1);    % randomly generated is okay
opts.maxit  = 100;              % we'll use this value of maxit later

% However, (0,0) or (1,1) are on the nonsmooth manifold and if GRANSO
% is started at either of them, it will break down on the first
% iteration.  This example highlights that it is imperative to start
% GRANSO at a point where the functions are smooth.

% Uncomment either of the following two lines to try starting GRANSO
% from (0,0) or (1,1), where the functions are not differentiable.

%     opts.x0 = ones(nvar,1);     % uncomment this line to try this point
%     opts.x0 = zeros(nvar,1);    % uncomment this line to try this point

% Uncomment the following two lines to try starting GRANSO from a
% uniformly perturbed version of (1,1).  pert_level needs to be at
% least 1e-3 or so to get consistently reliable optimization quality.

%     pert_level  = 1e-3;
%     opts.x0     = ones(nvar,1) + pert_level * (rand(nvar,1) - 0.5);


% NO NEED TO CHANGE ANYTHING BELOW


% SET UP THE (ANONYMOUS) "COMBINED" GRANSO FORMAT FUNCTION HANDLE

% First, we dynamically set the constant needed for the objective
parameters.w = 8;

% Embed the *current* value of w dynamically into the objective
% call mat2vec to enable GRANSO using matrix input
combined_fn = @(x) mat2vec(x,var_dim_map,nvar,parameters);



% SETUP THE LOGGING FEATURES

% Set up GRANSO's logging functions; pass opts.maxit to it so that
% storage can be preallocated for efficiency.
% The copy of makeHaltLogFunctions.m for this example is identical to
% one provided in the halt_log_template folder.
[halt_log_fn, get_log_fn]   = makeHaltLogFunctions(opts.maxit);

% Set GRANSO's logging function in opts
opts.halt_log_fn            = halt_log_fn;

% Call GRANSO using its "combined" format, with logging enabled.
opts.quadprog_opts.QPsolver = 'qpalm';
% opts.quadprog_opts.QPsolver = 'quadprog';

soln = granso(nvar,combined_fn,opts);

% GET THE HISTORY OF ITERATES
% Even if an error is thrown, the log generated until the error can be
% obtained by calling get_log_fn();
log     = get_log_fn();

end

