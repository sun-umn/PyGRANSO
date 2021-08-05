import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(grandparentdir)
from pygranso import pygranso
from private.mat2vec import mat2vec
from pygransoStruct import Options
import numpy as np
import scipy.io
import time

"""
  runExample: (example_mat/ex4)
      Run GRANSO on an eigenvalue optimization problem of a 
      static-output-feedback (SOF) plant:
      
          M = A + BXC,

      where A,B,C are all fixed real-valued matrices
          - A is n by n 
          - B is n by p
          - C is m by n
      and X is a real-valued p by m matrix of the optimization variables.

      The specific instance loaded when running this example has the
      following properties:
          - A,B,C were all generated via randn()
          - n = 200
          - p = 10
          - m = 20

      The objective is to minimize the maximum of the imaginary parts of
      the eigenvalues of M.  In other words, we want to restrict the
      spectrum of M to be contained in the smallest strip as possible
      centered on the x-axis (since the spectrum of M is symmetric with
      respect to the x-axis).

      The (inequality) constraint is that the system must be
      asymptotically stable, subject to a specified stability margin.  In
      other words, if the stability margin is 1, the spectral abscissa 
      must be at most -1.

      Read this source code.

  USAGE:
      soln = runExample();

  INPUT: [none]

  OUTPUT:
      soln        GRANSO's output struct


"""

# read input data from matlab file
file = currentdir + "/ex4_data_n=200.mat"
mat = scipy.io.loadmat(file)
mat_struct = mat['sys']
val = mat_struct[0,0]
A = val['A']
B = val['B']
p = B.shape[1]
C = val['C']
m = C.shape[0]

nvar        = m*p

# variable and corresponding dimensions
var_dim_map = {"XX": (p,m) }

# # calculate total number of scalar variables
# nvar = 0
# for dim in var_dim_map.values():
#     nvar = nvar + dim[0] * dim[1]

opts = Options()
opts.QPsolver = 'gurobi'
# opts.QPsolver = 'osqp'
opts.maxit = 200
opts.x0 = np.zeros((nvar,1))

feasibility_bias = False
if feasibility_bias:
    opts.steering_ineq_margin = np.inf    # default is 1e-6
    opts.steering_c_viol = 0.9         # default is 0.1
    opts.steering_c_mu = 0.1           # default is 0.9

stab_margin = 1

## FOR PLOTTING THE SPECTRUM



## SET UP THE ANONYMOUS FUNCTION HANDLE AND OPTIMIZE

start = time.time()
# call mat2vec to enable GRANSO using matrix input
combined_fn = lambda x: mat2vec(x,var_dim_map,nvar)
pygranso(nvar,combined_fn,opts)
end = time.time()
print(end - start)