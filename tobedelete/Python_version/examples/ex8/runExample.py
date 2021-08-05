import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(grandparentdir)
from pygranso import pygranso
from private.mat2vec import mat2vec
from pygransoStruct import Options
import numpy as np

"""
  runExample: (examples/ex5)
    Asl, Azam, and Michael L. Overton. "Analysis of the gradient method with 
    an Armijo?Wolfe line search on a class of non-smooth convex functions." 
    Optimization methods and software 35.2 (2020): 223-242.

"""

# variable and corresponding dimensions
var_dim_map = {"u": (10,1)}

# calculate total number of scalar variables
nvar = 0
for dim in var_dim_map.values():
    nvar = nvar + dim[0] * dim[1]



opts = Options()
opts.QPsolver = 'gurobi'
opts.maxit = 1000
opts.x0 = np.ones((10,1))
# opts.ngrad = 24

# call mat2vec to enable GRANSO using matrix input
combined_fn = lambda x: mat2vec(x,var_dim_map,nvar)

soln = pygranso(nvar,combined_fn,opts)

print(soln.final.x)
