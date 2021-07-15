function soln = runExample()
% runExample: (examples/ex5)
% Asl, Azam, and Michael L. Overton. "Analysis of the gradient method with 
% an Armijo?Wolfe line search on a class of non-smooth convex functions." 
% Optimization methods and software 35.2 (2020): 223-242.
        
% key
var = {'u'};
% value: dimension. e.g., 2 by 2 => [2,2]
dim = {[10,1]};
var_dim_map =  containers.Map(var, dim);

% calculate total number of scalar variables
nvar = 0;
for idx = 1:length(dim)
    curDim = dim(idx);
    nvar = nvar + curDim{1,1}(1)*curDim{1,1}(2);
end

% opts.quadprog_opts.QPsolver = 'qpalm';
opts.quadprog_opts.QPsolver = 'quadprog';
opts.x0 = ones(10,1);

 

%% call mat2vec to enable GRANSO using matrix input
combined_fn = @(x) mat2vec(x,var_dim_map,nvar);
soln = granso(nvar,combined_fn,opts);

 
    
end