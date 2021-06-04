function soln = runExample()

        
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


%% call mat2vec to enable GRANSO using matrix input
combined_fn = @(x) mat2vec(x,var_dim_map,nvar);
soln = granso(nvar,combined_fn);

 
    
end