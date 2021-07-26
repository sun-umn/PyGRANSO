import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(grandparentdir)
from pygranso import pygranso
from private.mat2vec import mat2vec_autodiff
from private.getNvar import getNvar
from pygransoStruct import Options
import numpy as np
import torch

# variable and corresponding dimensions
n = 1000
var_dim_map = {"x": (n,1)}

nvar = getNvar(var_dim_map)

opts = Options()
opts.QPsolver = 'gurobi'
opts.maxit = 10000
opts.x0 = 0.1*np.ones((nvar,1))
# opts.x0 = np.zeros((nvar,1))

# call mat2vec to enable GRANSO using matrix input
combined_fn = lambda x: mat2vec_autodiff(x,var_dim_map,nvar)
soln = pygranso(nvar,combined_fn,opts)


print(soln.final.x)


