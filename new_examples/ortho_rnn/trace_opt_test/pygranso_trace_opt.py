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
write_to_log = True

n = 10 # V: n*d
d = 5 # copnst: d*d

# folding_list = ['l2','l1','linf']
# folding_list = ['l2','l1']
folding_list = ['l2','l1','linf','unfolding']

total = 100 # total number of starting points
feasible_init = False
opt_tol = 1e-6
maxit = 50000
maxclocktime = 30
# QPsolver = "gurobi"
QPsolver = "osqp"
threshold = 0.99

square_flag = True
# square_flag = False

mu0 = 0.1
device = torch.device('cuda')
###############################################

if square_flag:
    square_str = "square_"
else:
    square_str = ""

maxfolding = ''
for str in folding_list:
    maxfolding = maxfolding + str + '_'

# save file
now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y_%H:%M:%S")

my_path = os.path.dirname(os.path.abspath(__file__))

if feasible_init:
    feasible = "feasible_init"
else:
    feasible = "gaussnormal_init"

name_str = "_n{}_d{}_{}_{}{}_total{}_maxtime{}".format(n,d,feasible,square_str,maxfolding,total,maxclocktime)

log_name = "log/" + date_time + name_str + '.txt'

print( name_str + "start\n\n")


if write_to_log:
    sys.stdout = open(os.path.join(my_path, log_name), 'w')



###################################################
torch.manual_seed(2023)
np.random.seed(2023)

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

#generate a list of markers and another of colors 
markers = [ "," , "o" , "v" , "^" , "<", ">", "." ]
colors = ['r','g','b','c','m', 'y', 'k']

idx = 0 # index for plots
for maxfolding in folding_list:
    print('\n\n\n'+maxfolding + '  start!')
    idx+=1

    def user_fn(X_struct,A,d):
        V = X_struct.V

        # objective function
        f = -torch.trace(V.T@A@V)

        # inequality constraint, matrix form
        ci = None

        # equality constraint
        # ce = None
        ce = pygransoStruct()

        constr_vec = (V.T@V - torch.eye(d).to(device=device, dtype=torch.double)).reshape(d**2,-1)

        if maxfolding == 'l1':
            # ce.c1 = norm(V.T@V - torch.eye(d).to(device=device, dtype=torch.double),1)
            ce.c1 = torch.sum(torch.abs(constr_vec))
        elif maxfolding == 'l2':
            # ce.c1 = norm(V.T@V - torch.eye(d).to(device=device, dtype=torch.double),2)
            ce.c1 = torch.sum(constr_vec**2)**0.5
        elif maxfolding == 'linf':
            # ce.c1 = norm(V.T@V - torch.eye(d).to(device=device, dtype=torch.double),float('inf'))
            ce.c1 = torch.amax(torch.abs(constr_vec))
        elif maxfolding == 'unfolding':
            ce.c1 = V.T@V - torch.eye(d).to(device=device, dtype=torch.double)
        else:
            print("Please specficy you maxfolding type!")
            exit()

        if square_flag:
            ce.c1 = ce.c1**2

        return [f,ci,ce]

    comb_fn = lambda X_struct : user_fn(X_struct,A,d)


    print("torch.trace(U.T@A@U) = {}".format(torch.trace(U.T@A@U)))
    print("sum of first d eigvals = {}".format(torch.sum(L[index[0:d]])))
    time_lst = []
    F_lst = []
    MF_lst = []
    termination_lst = []
    termination_lst_all = []
    TV_lst = []
    MF_TV_lst = []
    iter_lst = []


    start_loop = time.time()

    for i in range(total):
        print("i = {}".format(i))
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
        opts.mu0 = mu0
        opts.fvalquit = -torch.trace(U.T@A@U).item()*threshold


        if feasible_init:
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
            if soln.termination_code != 12 and soln.termination_code != 8:
                time_lst.append(end-start)
                F_lst.append(soln.final.f)
                MF_lst.append(soln.most_feasible.f)
                termination_lst.append(soln.termination_code)
                TV_lst.append(soln.final.tv) #total violation at x (vi + ve)
                MF_TV_lst.append(soln.most_feasible.tv)
                iter_lst.append(soln.iters)
            else:
                termination_lst_all.append("i = {}, termination code = {} ".format(i,soln.termination_code) )
        except Exception as e:
            print('skip pygranso')

        
    end_loop = time.time()
    print("Total Loop Wall Time: {}s".format(end_loop - start_loop))

    F_arr = np.array(F_lst)
    T_arr = np.array(time_lst)
    MF_arr = np.array(MF_lst)
    term_arr = np.array(termination_lst)
    TV_arr = np.array(TV_lst)
    MF_TV_arr = np.array(MF_TV_lst)
    iter_arr = np.array(iter_lst)

    index_sort = np.argsort(F_arr)
    index_sort = index_sort[::-1]
    sorted_F = F_arr[index_sort]
    sorted_T = T_arr[index_sort]
    sorted_MF = MF_arr[index_sort]
    sorted_termination = term_arr[index_sort]
    sorted_tv = TV_arr[index_sort]
    sorted_mf_tv = MF_TV_arr[index_sort]
    sorted_iter = iter_arr[index_sort]


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
    print("total violation tvi + tve = {}".format(sorted_tv))
    print("MF total violation tvi + tve = {}".format(sorted_mf_tv))
    print('iterations = {}'.format(sorted_iter))

    
    print(termination_lst_all)

    arr_len = sorted_F.shape[0]
    print("successful rate = {}".format(arr_len/total))
    plt.plot(np.arange(arr_len),sorted_F,color = colors[idx], marker = markers[idx], linestyle = '-',label=maxfolding)
    # plt.plot(np.arange(arr_len),sorted_MF,'go-',label='sorted_pygranso_sol_MF')

ana_sol = torch.trace(U.T@A@U).item()
plt.plot(np.arange(arr_len),np.array(arr_len*[-ana_sol]),color = colors[idx+1], linestyle = '-',label='analytical_sol')
plt.legend()

# plt.show()


png_title =  "png/sorted_F_" + date_time + name_str


plt.title(name_str)
plt.xlabel('sorted sample index')
plt.ylabel('obj_val')
plt.savefig(os.path.join(my_path, png_title))



if write_to_log:
    # end writing
    sys.stdout.close()
