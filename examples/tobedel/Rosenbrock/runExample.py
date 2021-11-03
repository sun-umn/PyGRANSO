import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso import pygranso
from pygransoStruct import Options, Data, GeneralStruct

# Please read the documentation on https://pygranso.readthedocs.io/en/latest/

# device = torch.device('cuda')
device = torch.device('cpu')

# variables and corresponding dimensions.
var_in = {"x1": (1,1), "x2": (1,1)}

# user defined options
opts = Options()
opts.QPsolver = 'osqp'
opts.maxit = 1000
opts.print_level = 1
opts.print_frequency = 1
opts.x0 = 0.5*torch.ones((2,1), device=device, dtype=torch.double)
opts.limited_mem_size = 1

def eval_obj(X_struct,data_in = None):
    # user defined variable, matirx form. torch tensor
    x1 = X_struct.x1
    x2 = X_struct.x2
    x1.requires_grad_(True)
    x2.requires_grad_(True)
    
    # objective function
    # obtain scalar from the torch tensor
    f = (8 * abs(x1**2 - x2) + (1 - x1)**2)[0,0]
    return f

def combinedFunction(X_struct,data_in = None):
    
    # user defined variable, matirx form. torch tensor
    x1 = X_struct.x1
    x2 = X_struct.x2
    x1.requires_grad_(True)
    x2.requires_grad_(True)
    
    # objective function
    # obtain scalar from the torch tensor
    f = (8 * abs(x1**2 - x2) + (1 - x1)**2)[0,0]

    # inequality constraint, matrix form
    ci = GeneralStruct()
    ci.c1 = (2**0.5)*x1-1  
    ci.c2 = 2*x2-1 

    # equality constraint 
    ce = None

    return [f,ci,ce]

comb_fn = lambda X_struct,data_in = None : combinedFunction(X_struct,data_in = None)
eval_fn = lambda X_struct,data_in = None : eval_obj(X_struct,data_in = None)


#  main algorithm  
start = time.time()
soln = pygranso(comb_fn,eval_fn,var_dim_map = var_in, torch_device=device, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))

print(soln.final.x)