function soln = runExample()
%   runExample: (example_mat/ex1_smoothNMF)
%       Run GRANSO on an smooth nonnegative matrix factorization optimization problem:
%
%           f(U,V) = || D - UV' ||_F^2,
%           s.t., U >= O, V >= O
%
%       where D is fixed real-valued matrices
%           - D is m by n
%       and U and V are real-valued matrix of the optimization variables
%           - U is m by k
%           - V is n by k
%
%       This example has the following properties:
%           - D = ones(m,n)
%           - m = 4
%           - n = 3
%           - k = 2
%
%       Read this source code.
%
%   USAGE:
%       soln = runExample();
%
%   INPUT: [none]
%
%   OUTPUT:
%       soln        GRANSO's output struct

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

%% call mat2vec to enable GRANSO using matrix input
combined_fn = @(x) mat2vec(x,var_dim_map,nvar);
soln = granso(nvar,combined_fn);


end