import time
from pyrsistent import b
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch
import torch.nn as nn
from torchvision.transforms import ToTensor

from pygranso.private.getObjGrad import getObjGradDL

from torch.linalg import norm

from sklearn import datasets



device = torch.device('cuda')



iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 2]
y = y[y != 2]

X /= X.max()  # Normalize X to speed-up convergence




X = torch.from_numpy(X).to(device=device, dtype=torch.double)
y = torch.from_numpy(y).to(device=device, dtype=torch.double)
[n,d] = X.shape
y = y.unsqueeze(1)

# variables and corresponding dimensions.
var_in = {"w": [d,1], "zeta": [n,1], "b": [n,1]}


def user_fn(X_struct,X,y):
    w = X_struct.w
    zeta = X_struct.zeta
    b = X_struct.b
    
    f = 0.5*w.T@w

    # inequality constraint 
    ci = pygransoStruct()
    constr = 1 - zeta - y*(X@w+b)
    ci.c1 = norm(constr,float('inf'))

    # equality constraint
    ce = None

    return [f,ci,ce]


comb_fn = lambda X_struct : user_fn(X_struct,X,y)

opts = pygransoStruct()
opts.torch_device = device

torch.manual_seed(1)
opts.x0 = torch.randn(2*n+d,1).to(device=device, dtype=torch.double)
# opts.opt_tol = 1e-6
# opts.viol_eq_tol = 1e-5
opts.maxit = 5000
# opts.fvalquit = 1e-6
opts.print_level = 1
opts.print_frequency = 100
# opts.print_ascii = True
# opts.limited_mem_size = 100
opts.double_precision = True

# opts.steering_c_viol = 0.02
opts.mu0 = 100

# opts.steering_c_mu = 0.95

# opts.globalAD = False # disable global auto-differentiation




# logits = model(inputs)
# _, predicted = torch.max(logits.data, 1)
# correct = (predicted == labels).sum().item()
# print("Initial acc = {:.2f}%".format((100 * correct/len(inputs))))


start = time.time()
soln = pygranso(var_spec = var_in,combined_fn = comb_fn,user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))


w = soln.final.x[0:d]
b = soln.final.x[d+n:d+2*n]
res = X@w+b
predicted = torch.zeros(n,1).to(device=device, dtype=torch.double)
predicted[res>=0.5] = 1
correct = (predicted == y).sum().item()
print("Final acc = {:.2f}%".format((100 * correct/n)))