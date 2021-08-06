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
d1 = 5
d2 = 10
var_in = {"M": (d1,d2),"S": (d1,d2)}

# parameters
parameters = Parameters()
torch.manual_seed(1)
parameters.eta = .5
parameters.Y = torch.randn(d1,d2)

# user defined options
opts = Options()
opts.QPsolver = 'osqp' 
opts.maxit = 1000
opts.x0 = np.ones((2*d1*d2,1))
opts.opt_tol = 1e-6
opts.fvalquit = 1e-6
opts.print_level = 1
opts.print_frequency = 10

#  main algorithm  
start = time.time()
soln = pygranso(var_in,parameters,opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))

# print(soln.final.x)
print(max(abs(soln.final.x))) # should be close to 1