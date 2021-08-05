import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(grandparentdir)
from pygranso import pygranso
from private.mat2vec import mat2vec
from pygransoStruct import Options
import numpy as np

# variable and corresponding dimensions
var_dim_map = {"U": (3,2), "V": (4,2)}

# calculate total number of scalar variables
nvar = 0
for dim in var_dim_map.values():
    nvar = nvar + dim[0] * dim[1]



opts = Options()
# opts.QPsolver = 'gurobi'
opts.QPsolver = 'osqp'
opts.maxit = 100
opts.x0 = np.ones((nvar,1))

# call mat2vec to enable GRANSO using matrix input
combined_fn = lambda x: mat2vec(x,var_dim_map,nvar)

# # test
# x = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1])
# [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec] = combined_fn(x)
# print(ci_grad_vec)
# print(f_grad_vec)

soln = pygranso(nvar,combined_fn,opts)
print(soln.final.x)
