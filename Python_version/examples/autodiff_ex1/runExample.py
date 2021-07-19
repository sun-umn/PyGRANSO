import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(grandparentdir)
from pygranso import pygranso
from mat2vec import mat2vec_autodiff
from pygransoStruct import Options
import numpy as np

"""
  runExample: (examples/ex1)
      Run GRANSO on a 2-variable nonsmooth Rosenbrock objective function,
      subject to simple bound constraints, with GRANSO's default
      parameters.
   
      Read this source code.
  
      This tutorial example shows:

          - how to call GRANSO using objective and constraint functions
            defined in combinedFunction.m 
      
          - how to set GRANSO's inputs when there aren't any 
            equality constraint functions (which also applies when there
            aren't any inequality constraints)

  USAGE:
      soln = runExample();

  INPUT: [none]
  
  OUTPUT:
      soln        GRANSO's output struct

  See also combinedFunction. 

"""

# variable and corresponding dimensions
var_dim_map = {"x1": (1,1), "x2": (1,1)}

# calculate total number of scalar variables
nvar = 0
for dim in var_dim_map.values():
    nvar = nvar + dim[0] * dim[1]



opts = Options()
opts.QPsolver = 'gurobi'
opts.maxit = 100
opts.x0 = np.array([0.51,0.51]).reshape((2,1))

# call mat2vec to enable GRANSO using matrix input
combined_fn = lambda x: mat2vec_autodiff(x,var_dim_map,nvar)

pygranso(nvar,combined_fn,opts)
