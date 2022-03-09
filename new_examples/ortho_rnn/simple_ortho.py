import time
import torch

import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')

from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

device = torch.device('cuda')
n = 5
d = 4
torch.manual_seed(0)

A = torch.randn(n,n)
A = A + A.T

# A = torch.eye(n)

# All the user-provided data (vector/matrix/tensor) must be in torch tensor format.
# As PyTorch tensor is single precision by default, one must explicitly set `dtype=torch.double`.
# Also, please make sure the device of provided torch tensor is the same as opts.torch_device.
A = A.to(device=device, dtype=torch.double)


L, U = torch.linalg.eig(A)

L = L.to(dtype=torch.double)
U = U.to(dtype=torch.double)

index = torch.argsort(L,descending=True)
# print(index)

# print(L)

# print(U)

U = U[:,index[0:d]]
# print(L[index])
# print(U.T@U)
# print(U)

# variables and corresponding dimensions.
var_in = {"V": [n,d]}


def user_fn(X_struct,A,d):
    V = X_struct.V

    # objective function
    f = -torch.trace(V.T@A@V)

    # inequality constraint, matrix form
    ci = None

    # equality constraint
    ce = pygransoStruct()
    ce.c1 = V.T@V - torch.eye(d).to(device=device, dtype=torch.double)

    return [f,ci,ce]

comb_fn = lambda X_struct : user_fn(X_struct,A,d)

opts = pygransoStruct()
opts.torch_device = device
opts.print_frequency = 1
# opts.x0 = .2 * torch.ones((2*d1*d2,1)).to(device=device, dtype=torch.double)
# opts.opt_tol = 1e-7
opts.maxit = 3000
# opts.mu0 = 10
opts.steering_c_viol = 0.02


start = time.time()
soln = pygranso(var_spec = var_in,combined_fn = comb_fn,user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))

V = torch.reshape(soln.final.x,(n,d))
# print(V)
# print(U)



rel_dist = torch.norm(V@V.T - U@U.T)/torch.norm(U@U.T)
# rel_dist = torch.norm(V@V.T - U@U.T)
print("torch.norm(V@V.T - U@U.T)/torch.norm(U@U.T) = {}".format(rel_dist))

# print(torch.norm(torch.trace(V.T@A@V)- torch.trace(U.T@A@U) ))

print("torch.trace(V.T@A@V) = {}".format(torch.trace(V.T@A@V)))
print("torch.trace(U.T@A@U) = {}".format(torch.trace(U.T@A@U)))
print("sum of first d eigvals = {}".format(torch.sum(L[index[0:d]])))

print("sorted eigs = {}".format(L[index]))

# print("U.T@U = {}".format(U.T@U))
# print("V.T@V = {}".format(V.T@V))