import time
from typing import Tuple
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from torch.linalg import norm
from scipy.stats import ortho_group
import numpy as np
from matplotlib import pyplot as plt 
import os
from datetime import datetime

###############################################
n = 10 # V: n*d
d = 5 # copnst: d*d
# maxfolding = 'unfolding'
maxfolding = 'l2'
total = 500 # total number of starting points
feasible_init = False
opt_tol = 1e-6
maxit = 1000
maxclocktime = 20
# QPsolver = "gurobi"
QPsolver = "osqp"
###############################################

device = torch.device('cuda')
torch.manual_seed(2023)

A = torch.randn(n,n)
A = (A + A.T)/2
# All the user-provided data (vector/matrix/tensor) must be in torch tensor format.
# As PyTorch tensor is single precision by default, one must explicitly set `dtype=torch.double`.
# Also, please make sure the device of provided torch tensor is the same as opts.torch_device.
A = A.to(device=device, dtype=torch.double)

L, U = torch.linalg.eig(A)
L = L.to(dtype=torch.double)
U = U.to(dtype=torch.double)
index = torch.argsort(L,descending=True)
U = U[:,index[0:d]]

# variables and corresponding dimensions.
var_in = {"V": [n,d]}


def user_fn(X_struct,A,d):
    V = X_struct.V

    # objective function
    f = -torch.trace(V.T@A@V)

    # inequality constraint, matrix form
    ci = None

    # equality constraint
    # ce = None
    ce = pygransoStruct()

    if maxfolding == 'l1':
        ce.c1 = norm(V.T@V - torch.eye(d).to(device=device, dtype=torch.double),1)
    elif maxfolding == 'l2':
        ce.c1 = norm(V.T@V - torch.eye(d).to(device=device, dtype=torch.double),2)
    elif maxfolding == 'linf':
        ce.c1 = norm(V.T@V - torch.eye(d).to(device=device, dtype=torch.double),float('inf'))
    elif maxfolding == 'unfolding':
        ce.c1 = V.T@V - torch.eye(d).to(device=device, dtype=torch.double)
    else:
        print("Please specficy you maxfolding type!")
        exit()

    return [f,ci,ce]

comb_fn = lambda X_struct : user_fn(X_struct,A,d)

# np.random.seed(2022)
# x = ortho_group.rvs(n)
# x = x[:,0:d].reshape(-1,1)
# opts.x0 = torch.from_numpy(x).to(device=device, dtype=torch.double) + eps*torch.randn((n*d,1)).to(device=device, dtype=torch.double)

# opts.maxit = 5000
# opts.mu0 = 1.3e-2
# opts.steering_c_viol = 0.02
# opts.limited_mem_size = 100


print("torch.trace(U.T@A@U) = {}".format(torch.trace(U.T@A@U)))
print("sum of first d eigvals = {}".format(torch.sum(L[index[0:d]])))
time_lst = []
F_lst = []
MF_lst = []
termination_lst = []


start_loop = time.time()

for i in range(total):
    print("i = {}".format(i))
    torch.manual_seed(2022+i)
    opts = pygransoStruct()
    opts.torch_device = device
    opts.print_frequency = 10
    opts.maxit = maxit
    opts.print_use_orange = False
    opts.print_ascii = True
    opts.quadprog_info_msg  = False
    opts.opt_tol = opt_tol
    opts.maxclocktime = maxclocktime
    opts.QPsolver = QPsolver


    if feasible_init:
        np.random.seed(2022+i)
        x = ortho_group.rvs(n)
        x = x[:,0:d].reshape(-1,1)
        eps = 1e-5
        opts.x0 = torch.from_numpy(x).to(device=device, dtype=torch.double) + eps*torch.randn((n*d,1)).to(device=device, dtype=torch.double)
    else:
        opts.x0 =  torch.randn((n*d,1)).to(device=device, dtype=torch.double)
        opts.x0 = opts.x0/norm(opts.x0)

    try:
        start = time.time()
        soln = pygranso(var_spec = var_in,combined_fn = comb_fn,user_opts = opts)
        end = time.time()
        print("Total Wall Time: {}s".format(end - start))
        if soln.termination_code != 5 and soln.termination_code != 12 and soln.termination_code != 8:
            time_lst.append(end-start)
            F_lst.append(soln.final.f)
            MF_lst.append(soln.most_feasible.f)
            termination_lst.append(soln.termination_code)
    except Exception as e:
        print('skip pygranso')

    
end_loop = time.time()
print("Total Loop Wall Time: {}s".format(end_loop - start_loop))

F_arr = np.array(F_lst)
T_arr = np.array(time_lst)
MF_arr = np.array(MF_lst)
term_arr = np.array(termination_lst)

index_sort = np.argsort(F_arr)
index_sort = index_sort[::-1]
sorted_F = F_arr[index_sort]
sorted_T = T_arr[index_sort]
sorted_MF = MF_arr[index_sort]
sorted_termination = term_arr[index_sort]

V = torch.reshape(soln.final.x,(n,d))
rel_dist = torch.norm(V@V.T - U@U.T)/torch.norm(U@U.T)

print("torch.norm(V@V.T - U@U.T)/torch.norm(U@U.T) = {}".format(rel_dist))
print("torch.trace(V.T@A@V) = {}".format(torch.trace(V.T@A@V)))
print("torch.trace(U.T@A@U) = {}".format(torch.trace(U.T@A@U)))
print("sum of first d eigvals = {}".format(torch.sum(L[index[0:d]])))

print("sorted eigs = {}".format(L[index]))

print( "Time = {}".format(sorted_T) )
print("F obj = {}".format(sorted_F))
print("MF obj = {}".format(sorted_MF))
print("termination code = {}".format(sorted_termination))

arr_len = sorted_F.shape[0]
plt.plot(np.arange(arr_len),sorted_F,'ro-',label='sorted_pygranso_sol_F')
plt.plot(np.arange(arr_len),sorted_MF,'go-',label='sorted_pygranso_sol_MF')

ana_sol = torch.trace(U.T@A@U).item()
plt.plot(np.arange(arr_len),np.array(arr_len*[-ana_sol]),'b-',label='analytical_sol')
plt.legend()
# plt.show()

now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y_%H:%M:%S")
if feasible_init:
    feasible = "feasible_init"
else:
    feasible = "gaussnormal_init"


png_title =  "png/sorted_F_" + date_time + "_n{}_d{}_{}_{}".format(n,d,feasible,maxfolding)

my_path = os.path.dirname(os.path.abspath(__file__))

plt.savefig(os.path.join(my_path, png_title))

print("successful rate = {}".format(arr_len/total))