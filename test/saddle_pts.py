import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso import pygranso
from pygransoStruct import pygransoStruct

device = torch.device('cuda')
# variables and corresponding dimensions.
var_in = {"x": [1], "y": [1]}

def comb_fn(X_struct):
    x = X_struct.x
    y = X_struct.y

    # objective function
    f = x**2 - y**2

    # inequality constraint, matrix form
    ci = pygransoStruct()
    ci.c1 = torch.abs(y)-1

    # ci = None

    # equality constraint
    ce = None

    return [f,ci,ce]

opts = pygransoStruct()

# opts.x0 = torch.ones((2,1), device=device, dtype=torch.double) * 1e-1

# opts.x0 = torch.zeros((2,1), device=device, dtype=torch.double) 

x0 = torch.FloatTensor([6e-3,2e-3]).to(device=device, dtype=torch.double)
opts.x0 = torch.unsqueeze(x0, 1)

# x0 = torch.FloatTensor([0,1]).to(device=device, dtype=torch.double)
# opts.x0 = torch.unsqueeze(x0, 1)

opts.torch_device = device
opts.mu0 = 0.1

start = time.time()
soln = pygranso(var_spec = var_in,combined_fn = comb_fn, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))
print(soln.final.x)