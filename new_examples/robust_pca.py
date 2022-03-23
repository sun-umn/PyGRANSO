import time
import torch

import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')

from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

from torch.linalg import norm


device = torch.device('cuda')
d1 = 10
d2 = 10
torch.manual_seed(1)
eta = .05
# All the user-provided data (vector/matrix/tensor) must be in torch tensor format.
# As PyTorch tensor is single precision by default, one must explicitly set `dtype=torch.double`.
# Also, please make sure the device of provided torch tensor is the same as opts.torch_device.
Y = torch.randn(d1,d2).to(device=device, dtype=torch.double)



# variables and corresponding dimensions.
var_in = {"M": [d1,d2],"S": [d1,d2]}


def user_fn(X_struct,Y):
    M = X_struct.M
    S = X_struct.S

    # objective function
    f = torch.norm(M, p = 'nuc') + eta * torch.norm(S, p = 1)

    # inequality constraint, matrix form
    ci = None

    # equality constraint
    ce = pygransoStruct()
    # ce.c1 = M + S - Y
    ce.c1 = norm(M+S-Y,float('inf'))

    return [f,ci,ce]

comb_fn = lambda X_struct : user_fn(X_struct,Y)

opts = pygransoStruct()
opts.torch_device = device
opts.print_frequency = 100
# opts.x0 = .2 * torch.ones((2*d1*d2,1)).to(device=device, dtype=torch.double)
torch.manual_seed(1)
opts.x0 = torch.randn((2*d1*d2,1)).to(device=device, dtype=torch.double)
# opts.opt_tol = 1e-6
opts.maxit = 2000

start = time.time()
soln = pygranso(var_spec = var_in,combined_fn = comb_fn,user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))