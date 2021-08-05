import time
import numpy as np
import torch
import numpy.linalg as la
from scipy.stats import norm
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append(r'C:\Users\Buyun\Documents\GitHub\PyGRANSO')
from pygranso import pygranso
from pygransoStruct import Options, Parameters

###################### variable and corresponding dimensions. ##########################
########################################################################################
# Use a python dictionary to indicate variable name and corresponding matrix dimension
# If variable is a scalar, dimension is (1,1)
# If variable is a vector, dimension is (n,1)
# If variable is a matrix, dimension is (n1,n2)
n = 30
var_dim_map = {"q": (n,1)}

########################################################################################
###################### Structure to specify the options in PyGRANSO ####################
opts = Options()
# Quadratic Programming solver can be specified by the user, the default is osqp 
opts.QPsolver = 'osqp' # opts.QPsolver = 'gurobi'
# maximum iterations. Default is 1000
opts.maxit = 10000
# User defined initialization. Default is random norm distribution
# x0 should be numpy array
np.random.seed(1)
x0 = norm.ppf(np.random.rand(n,1))
x0 /= la.norm(x0,2)
opts.x0 = x0
# Tolerance for reaching (approximate) optimality/stationarity
opts.opt_tol = 1e-6
# Quit if objective function drops below this value at a feasible 
# iterate (that is, satisfying feasibility tolerances 
# opts.viol_ineq_tol and opts.viol_eq_tol).
opts.fvalquit = 1e-6
#    User defined print_level. Default is 1
#    Level of detail printed to console regarding optimization progress:
#    0 - no printing whatsoever
#    1 - prints info for each iteration  
opts.print_level = 1
# Sets how often the iterations are printed. Default is 1. When set to one, every
# iteration is printed; when set to 10, only every 10th iteration is
# printed.  When set to inf, no iterations are printed, except for
# at x0.  Note that this only affects .print_level == 1 printing;
# all messages from higher values of .print_level will still be
# printed no matter what iteration they occurred on.
opts.print_frequency = 10

########################################################################################
###########################  structure for parameters  #################################

m = 10*n**2   # sample complexity
theta = 0.3   # sparsity level
# structure for parameters
# All parameters used in objective and constraint functions are 
# suggested to be specified here to save computational sources
# all non-scalaer parameters should be Pytorch tensor
parameters = Parameters()
Y = norm.ppf(np.random.rand(n,m)) * (norm.ppf(np.random.rand(n,m)) <= theta)
parameters.Y = torch.from_numpy(Y) 
parameters.m = m

########################################################################################
###########################  main algorithm  ###########################################

start = time.time()
soln = pygranso(var_dim_map,parameters,opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))

# print(soln.final.x)
print(max(abs(soln.final.x))) # should be close to 1