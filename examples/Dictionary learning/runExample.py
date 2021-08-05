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

# Please read the documentation on https://pygranso.readthedocs.io/en/latest/

# variables and corresponding dimensions.
n = 30
vars = {"q": (n,1)}

# parameters
parameters = Parameters()
m = 10*n**2   # sample complexity
theta = 0.3   # sparsity level
Y = norm.ppf(np.random.rand(n,m)) * (norm.ppf(np.random.rand(n,m)) <= theta)  # Bernoulli-Gaussian model
parameters.Y = torch.from_numpy(Y) 
parameters.m = m


# user defined options
opts = Options()
opts.QPsolver = 'osqp' 
opts.maxit = 10000
# User defined initialization. 
np.random.seed(1)
x0 = norm.ppf(np.random.rand(n,1))
x0 /= la.norm(x0,2)
opts.x0 = x0
opts.opt_tol = 1e-6
opts.fvalquit = 1e-6
opts.print_level = 1
opts.print_frequency = 10



########################################################################################
###########################  main algorithm  ###########################################

start = time.time()
soln = pygranso(vars,parameters,opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))

# print(soln.final.x)
print(max(abs(soln.final.x))) # should be close to 1