import time
import numpy as np
import torch
import numpy.linalg as la
from scipy.stats import norm
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append(r'C:\Users\Buyun\Documents\GitHub\PyGRANSO')
from pygranso import pygranso
from pygransoStruct import Options, Data

# Please read the documentation on https://pygranso.readthedocs.io/en/latest/
device = torch.device('cuda' )
# device = torch.device('cpu' )


# variables and corresponding dimensions.
n = 200
var_in = {"q": (n,1)}

# data_in
data_in = Data()
m = 10*n**2   # sample complexity
theta = 0.3   # sparsity level
Y = norm.ppf(np.random.rand(n,m)) * (norm.ppf(np.random.rand(n,m)) <= theta)  # Bernoulli-Gaussian model
data_in.Y = torch.from_numpy(Y).to(device=device, dtype=torch.double)
data_in.m = m

# user defined options
opts = Options()
opts.QPsolver = 'osqp' 
opts.maxit = 500
# User defined initialization. 
np.random.seed(1)
x0 = norm.ppf(np.random.rand(n,1))
x0 /= la.norm(x0,2)
x0 = torch.from_numpy(x0).to(device=device, dtype=torch.double)
opts.x0 = x0
opts.opt_tol = 1e-6
opts.fvalquit = 1e-6
opts.print_level = 1
opts.print_frequency = 10
opts.print_ascii = True

#  main algorithm  
start = time.time()
soln = pygranso(var_dim_map = var_in,user_data = data_in, user_opts = opts, torch_device=device)
end = time.time()
print("Total Wall Time: {}s".format(end - start))

# print(soln.final.x)
print(max(abs(soln.final.x))) # should be close to 1