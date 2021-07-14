from pygranso import pygranso
from mat2vec import mat2vec
from pygransoStruct import Options
import numpy as np



# help(pygranso)

# variable and corresponding dimensions
var_dim_map = {"U": (3,2), "V": (4,2)}

# calculate total number of scalar variables
nvar = 0
for dim in var_dim_map.values():
    nvar = nvar + dim[0] * dim[1]



opts = Options()
opts.QPsolver = 'gurobi'
opts.maxit = 3
opts.x0 = np.ones((nvar,1))

# call mat2vec to enable GRANSO using matrix input
combined_fn = lambda x: mat2vec(x,var_dim_map,nvar)

# # test
# x = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1])
# [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec] = combined_fn(x)
# print(ci_grad_vec)
# print(f_grad_vec)

pygranso(nvar,combined_fn,opts)
