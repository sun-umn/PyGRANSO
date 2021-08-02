function soln = runExample()


rng(12);
theta = .3;   % sparsity level
n = 100;   % dimension
m = round(10*n^2);    % number of measurements

var = {'q'};
% value: dimension. e.g., 2 by 2 => [2,2]
dim = {[n,1]};
var_dim_map =  containers.Map(var, dim);

% calculate total number of scalar variables
nvar = 0;
for idx = 1:length(dim)
    curDim = dim(idx);
    nvar = nvar + curDim{1,1}(1)*curDim{1,1}(2);
end

% opts.quadprog_opts.QPsolver = 'qpalm';
% opts.quadprog_opts.QPsolver = 'quadprog';
opts.quadprog_opts.QPsolver = 'gurobi';

opts.x0 = ones(n,1);
opts.maxit = 2000;
opts.opt_tol = 1e-6;
%opts.limited_mem_size = 250;   % lbfgs
opts.fvalquit = 1e-6;
opts.print_level = 1;
opts.print_frequency = 10; 



parameters.Y = randn(n,m) .* (rand(n,m) <= theta);   % Bernoulli-Gaussian model
parameters.m = m;

tic
combined_fn = @(x) mat2vec(x,var_dim_map, nvar,parameters );
soln = granso(nvar,combined_fn,opts);
toc 

max(abs(soln.final.x))   % should be close to 1
 
    
end