function soln = runExample()
%   runExample: (examples/ex5)
%       Run GRANSO on a 2-variable nonsmooth Rosenbrock objective function,
%       subject to simple bound constraints, with GRANSO's default
%       parameters.
%    
%       Read this source code.
%   
%       This tutorial example shows:
%
%           - how to call GRANSO using objective and constraint functions
%             defined in .m files 
%       
%           - how to set GRANSO's inputs when there aren't any 
%             equality constraint functions (which also applies when there
%             aren't any inequality constraints)
%
%   USAGE:
%       soln = runExample();
% 
%   INPUT: [none]
%   
%   OUTPUT:
%       soln        GRANSO's output struct
%
%   See also objectiveFunction, inequalityConstraint. 

%% specify input variables 
% key: input variables
var = {'U','V'};
% value: dimension. e.g., 3 by 2 => [3,2]
dim = {[3,2],[4,2]};
var_dim_map =  containers.Map(var, dim);

% calculate total number of scalar variables
nvar = 0;
for idx = 1:length(dim)
    curDim = dim(idx);
    nvar = nvar + curDim{1,1}(1)*curDim{1,1}(2);
end
opts.quadprog_opts.QPsolver = 'qpalm';
% opts.quadprog_opts.QPsolver = 'quadprog';

%% call mat2vec to enable GRANSO using matrix input
combined_fn = @(x) mat2vec(x,var_dim_map,nvar);
soln = granso(nvar,combined_fn,opts);


end