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
n = 80
var_in = {"x": (n,1)}

# parameters
parameters = Parameters()

eta = 0.5 # parameter for penalty term
torch.manual_seed(1)
b = torch.rand(n,1)
pos_one = torch.ones(n-1)
neg_one = -torch.ones(n-1)
F = torch.zeros(n-1,n)
F[:,0:n-1] += torch.diag(neg_one,0) 
F[:,1:n] += torch.diag(pos_one,0)
parameters.F = F.double()  # double precision requireed in torch operations 
parameters.b = b
parameters.eta = np.double(eta) # double precision requireed in torch operations 

# user defined options
opts = Options()
opts.QPsolver = 'osqp' 
opts.maxit = 1000
opts.x0 = np.ones((n,1))
opts.print_level = 1
opts.print_frequency = 10

#  main algorithm  
start = time.time()
soln = pygranso(var_in,parameters,opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))

# print(soln.final.x)