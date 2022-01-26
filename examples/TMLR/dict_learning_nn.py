import time
import torch
import sys
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct 
from pygranso.private.getNvar import getNvarTorch
import torch.nn as nn
import numpy.linalg as la
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import torch
from scipy.stats import norm

device = torch.device('cuda')

class Dict_Learning(nn.Module):
    
    def __init__(self,n):
        super().__init__()
        np.random.seed(1)
        q0 = norm.ppf(np.random.rand(n,1))
        q0 /= la.norm(q0,2)
        self.q = nn.Parameter( torch.from_numpy(q0) )
    
    def forward(self, Y,m):
        qtY = self.q.T @ Y
        f = 1/m * torch.norm(qtY, p = 1)
        return f

## Data initialization
n = 30
np.random.seed(1)
m = 10*n**2   # sample complexity
theta = 0.3   # sparsity level
Y = norm.ppf(np.random.rand(n,m)) * (norm.ppf(np.random.rand(n,m)) <= theta)  # Bernoulli-Gaussian model
# All the user-provided data (vector/matrix/tensor) must be in torch tensor format.
# As PyTorch tensor is single precision by default, one must explicitly set `dtype=torch.double`.
# Also, please make sure the device of provided torch tensor is the same as opts.torch_device.
Y = torch.from_numpy(Y).to(device=device, dtype=torch.double)

torch.manual_seed(0)

model = Dict_Learning(n).to(device=device, dtype=torch.double)

def user_fn(model,Y,m):
    # objective function    
    f = model(Y,m)

    # q = model.state_dict()['q']
    # q.requires_grad_(True)
    q = list(model.parameters())[0]
    # for parameter in model.parameters():
    #     q = parameter

    # inequality constraint
    ci = None

    # equality constraint 
    ce = pygransoStruct()
    ce.c1 = q.T @ q - 1
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # ce = None
    # print("ce = {}".format(ce.c1.item()))
    # print("q = {}".format(q))
    return [f,ci,ce]

comb_fn = lambda model : user_fn(model,Y,m)

opts = pygransoStruct()
opts.torch_device = device
nvar = getNvarTorch(model.parameters())
opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
# opts.opt_tol = 1e-5
opts.maxit = 10000
# opts.fvalquit = 1e-6
opts.print_level = 1
opts.print_frequency = 1

start = time.time()
soln = pygranso(var_spec= model, combined_fn = comb_fn, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))