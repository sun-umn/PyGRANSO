import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(grandparentdir)
from pygranso import pygranso
from private.mat2vec import mat2vec_autodiff
from private.getNvar import getNvar
from pygransoStruct import Options, general_struct
import time
import numpy as np
import torch
import numpy.linalg as la
import scipy.io
from scipy.stats import norm




# variable and corresponding dimensions
n = 300
m = 10*n**2
var_dim_map = {"q": (n,1)}

nvar = getNvar(var_dim_map)

opts = Options()
# opts.QPsolver = 'gurobi'
opts.QPsolver = 'osqp'
opts.maxit = 10000
# opts.maxit = 100

seed_num = 1
print( "seed_num = %d" %seed_num)
np.random.seed(seed_num)
x0 = norm.ppf(np.random.rand(n,1))
x0 = x0/la.norm(x0,2)
opts.x0 = x0
print(x0.T@x0)

# print( opts.x0.T @ opts.x0 )
# opts.x0 = 0.1*np.ones((nvar,1))

opts.opt_tol = 1e-6
opts.fvalquit = 1e-6
opts.print_level = 1
opts.print_frequency = 10


theta = 0.3   # sparsity level
parameters = general_struct()
Y = norm.ppf(np.random.rand(n,m)) * (norm.ppf(np.random.rand(n,m)) <= theta)
# print(Y)
parameters.Y = torch.from_numpy(Y)


parameters.m = m

start = time.time()
# call mat2vec to enable GRANSO using matrix input
combined_fn = lambda x: mat2vec_autodiff(x,var_dim_map,nvar,parameters)
soln = pygranso(nvar,combined_fn,opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))


# print(soln.final.x)
print(max(abs(soln.final.x))) # should be close to 1