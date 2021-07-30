function soln = runExample()

var = {'q'};
% value: dimension. e.g., 2 by 2 => [2,2]
dim = {[100,1]};
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

opts.x0 = ones(n);
opts.maxit = 2000;
opts.opt_tol = 1e-6;
%opts.limited_mem_size = 250;   % lbfgs
opts.fvalquit = 1e-6;
opts.print_level = 1;
opts.print_frequency = 10; 

tic;
soln = granso(n, obj, [], ec, opts);   % randomly initialize
toc


max(abs(soln.final.x))   % should be close to 1
%stem(soln.final.x)


%% call mat2vec to enable GRANSO using matrix input
combined_fn = @(x) mat2vec(x,var_dim_map,nvar);
soln = granso(nvar,combined_fn,opts);



 
    
end