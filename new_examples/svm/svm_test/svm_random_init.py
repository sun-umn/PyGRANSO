from functools import partial
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
from sklearn import datasets


###############################################
write_to_log = True

# maxfolding = 'unfolding'
maxfolding = 'l2'
# maxfolding = 'l1'
# maxfolding = 'linf'

data_iris = False

total = 1 # total number of starting points
opt_tol = 1e-6
maxit = 120000
maxclocktime = 5400
# QPsolver = "gurobi"
QPsolver = "osqp"

# square_flag = True
square_flag = False


partial_data = True
dp_num = 100

zeta = 0.4

###############################################


# save file
now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y_%H:%M:%S")

my_path = os.path.dirname(os.path.abspath(__file__))

if data_iris:
    data_name = 'iris'
else:
    data_name = 'bc'

if partial_data:
    data_num = "dp_num_{}".format(dp_num)
else:
    data_num = ""

if square_flag:
    square_str = "square "
else:
    square_str = ""

log_name = "log/" + date_time + "{}_{}{}_total{}_zeta{}.txt".format(data_name,square_str,maxfolding,total,int(zeta*100)) + data_num



print("{}_{}{}_total{}_zeta{} start\n\n".format(data_name,square_str,maxfolding,total,int(zeta*100)))


if write_to_log:
    sys.stdout = open(os.path.join(my_path, log_name), 'w')



###################################################
device = torch.device('cuda')
torch.manual_seed(2023)
# np.random.seed(2023)

if data_iris:
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = X[y != 2]
    y = y[y != 2]
    y[y==0] = -1

    X /= X.max()  # Normalize X to speed-up convergence
else:
    bc = datasets.load_breast_cancer()
    X = bc.data
    y = bc.target
    if partial_data:
        X = X[0:dp_num]
        y = y[0:dp_num]
    y[y==0] = -1
    X /= X.max()  # Normalize X to speed-up convergence


X = torch.from_numpy(X).to(device=device, dtype=torch.double)
y = torch.from_numpy(y).to(device=device, dtype=torch.double)
[n,d] = X.shape
y = y.unsqueeze(1)



# variables and corresponding dimensions.
var_in = {"w": [d,1], "b": [1,1]}

def user_fn(X_struct,X,y, zeta):
    w = X_struct.w
    b = X_struct.b    
    f = 0.5*w.T@w 
    # inequality constraint 
    ci = pygransoStruct()
    constr = 1 - zeta - y*(X@w+b)
    constr = constr.to(device=device, dtype=torch.double)

    if maxfolding == 'l1':
        ci.c1 = torch.sum(torch.clamp(constr, min=0)) # l1
    elif maxfolding == 'l2':
        ci.c1 = torch.sum(torch.clamp(constr, min=0)**2)**0.5 # l2
    elif maxfolding == 'linf':
        ci.c1 = torch.max(constr) # l_inf
    elif maxfolding == 'unfolding':
        ci.c1 = constr
    else:
        print("Please specficy you maxfolding type!")
        exit()

    if square_flag:
        ci.c1 = ci.c1**2

    # equality constraint
    ce = None

    return [f,ci,ce]

comb_fn = lambda X_struct : user_fn(X_struct,X,y,zeta)



time_lst = []
F_lst = []
MF_lst = []
termination_lst = []
termination_lst_all = []
acc_lst = []
TV_lst = []
MF_TV_lst = []


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


    opts.x0 =  torch.randn((d+1,1)).to(device=device, dtype=torch.double)
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
            w = soln.final.x[0:d]
            b = soln.final.x[d:d+1]
            res = X@w+b
            predicted = torch.zeros(n,1).to(device=device, dtype=torch.double)
            predicted[res>=0] = 1
            predicted[res<0] = -1
            correct = (predicted == y).sum().item()
            acc = correct/n
            print("Final acc = {:.2f}%".format((100 * acc)))
            acc_lst.append(acc)
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
acc_arr = np.array(acc_lst)
TV_arr = np.array(TV_lst)
MF_TV_arr = np.array(MF_TV_lst)

index_sort = np.argsort(F_arr)
index_sort = index_sort[::-1]
sorted_F = F_arr[index_sort]
sorted_T = T_arr[index_sort]
sorted_MF = MF_arr[index_sort]
sorted_termination = term_arr[index_sort]
sorted_acc = acc_arr[index_sort]
sorted_tv = TV_arr[index_sort]
sorted_mf_tv = MF_TV_arr[index_sort]

print( "Time = {}".format(sorted_T) )
print("F obj = {}".format(sorted_F))
print("MF obj = {}".format(sorted_MF))
print("termination code = {}".format(sorted_termination))
print("train acc = {}".format(sorted_acc))
print("total violation tvi + tve = {}".format(sorted_tv))
print("MF total violation tvi + tve = {}".format(sorted_mf_tv))


arr_len = sorted_F.shape[0]
plt.plot(np.arange(arr_len),sorted_F,'ro-',label='sorted_pygranso_sol_F')
plt.plot(np.arange(arr_len),sorted_MF,'go-',label='sorted_pygranso_sol_MF')

plt.legend()

# plt.show()

png_title =  "png/" + date_time + "_{}_{}{}_total{}_zeta{}".format(data_name,square_str,maxfolding,total,int(zeta*100)) + data_num


plt.title("{}_{}{}_total{}_zeta{}".format(data_name,square_str,maxfolding,total,int(zeta*100)) + data_num )
plt.xlabel('sorted sample index')
plt.ylabel('obj_val')
plt.savefig(os.path.join(my_path, png_title))

print("successful rate = {}".format(arr_len/total))
print(termination_lst_all)

if write_to_log:
    # end writing
    sys.stdout.close()
