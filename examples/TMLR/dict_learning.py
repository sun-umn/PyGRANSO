import time
import numpy as np
import torch
import numpy.linalg as la
from scipy.stats import norm
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

device = torch.device('cuda')
n = 30

np.random.seed(1)
m = 10*n**2   # sample complexity
theta = 0.3   # sparsity level
Y = norm.ppf(np.random.rand(n,m)) * (norm.ppf(np.random.rand(n,m)) <= theta)  # Bernoulli-Gaussian model
# All the user-provided data (vector/matrix/tensor) must be in torch tensor format.
# As PyTorch tensor is single precision by default, one must explicitly set `dtype=torch.double`.
# Also, please make sure the device of provided torch tensor is the same as opts.torch_device.
Y = torch.from_numpy(Y).to(device=device, dtype=torch.double)

# variables and corresponding dimensions.
var_in = {"q": [n,1]}


def user_fn(X_struct,Y):
    q = X_struct.q

    # objective function
    qtY = q.T @ Y
    f = 1/m * torch.norm(qtY, p = 1)

    # inequality constraint, matrix form
    ci = None

    # # equality constraint
    ce = pygransoStruct()
    ce.c1 = q.T @ q - 1
    # ce = None
    # print("ce = {}".format(ce.c1.item()))
    return [f,ci,ce]

comb_fn = lambda X_struct : user_fn(X_struct,Y)

opts = pygransoStruct()
opts.torch_device = device
opts.maxit = 1000
np.random.seed(1)
x0 = norm.ppf(np.random.rand(n,1))
x0 /= la.norm(x0,2)
opts.x0 = torch.from_numpy(x0).to(device=device, dtype=torch.double)

opts.print_frequency = 1

start = time.time()
soln = pygranso(var_spec = var_in,combined_fn = comb_fn,user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))
print(max(abs(soln.final.x))) # should be close to 1