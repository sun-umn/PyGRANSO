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
from sklearn.preprocessing import normalize

###############################################
write_to_log = True


# folding_list = ['l2','l1']

folding_list = ['l2','l1','linf','unfolding']


# data_name = 'iris'
# data_name = 'bc' # breast cancer 
data_name = 'lfw_pairs' # large dataset


total = 10 # total number of starting points
opt_tol = 1e-6
maxit = 120000
maxclocktime = 60
# QPsolver = "gurobi"
QPsolver = "osqp"

# square_flag = True
square_flag = False


partial_data = False
dp_num = 100

zeta = 0.0

###############################################
if square_flag:
    square_str = "square "
else:
    square_str = ""


if partial_data:
    data_num = "dp_num_{}".format(dp_num)
else:
    data_num = ""

maxfolding = ''
for str in folding_list:
    maxfolding = maxfolding + str + '_'

name_str = "{}_{}{}_total{}_zeta{}_maxtime{}".format(data_name,square_str,maxfolding,total,int(zeta*100),maxclocktime) + data_num

# save file
now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y_%H:%M:%S")
my_path = os.path.dirname(os.path.abspath(__file__))
log_name = "log/" + date_time + name_str + ".txt"
print( name_str + "start\n\n")
if write_to_log:
    sys.stdout = open(os.path.join(my_path, log_name), 'w')

###################################################
device = torch.device('cuda')
torch.manual_seed(2023)

if data_name == 'iris':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = X[y != 2]
    y = y[y != 2]
    y[y==0] = -1

    X /= X.max()  # Normalize X to speed-up convergence
elif data_name == 'bc':
    bc = datasets.load_breast_cancer()
    X = bc.data
    y = bc.target
    if partial_data:
        X = X[0:dp_num]
        y = y[0:dp_num]
    y[y==0] = -1
    X = normalize(X,axis=0)  # Normalize X to speed-up convergence

elif data_name == 'lfw_pairs':
    lfw_pairs = datasets.fetch_lfw_pairs(subset='train')
    X = lfw_pairs.data
    y = lfw_pairs.target
    names = lfw_pairs.target_names
    if partial_data:
        X = X[0:dp_num]
        y = y[0:dp_num]
    y[y==0] = -1
    X = normalize(X,axis=0)  # Normalize X to speed-up convergence
else:
    print('please specify a legal data name')

X = torch.from_numpy(X).to(device=device, dtype=torch.double)
y = torch.from_numpy(y).to(device=device, dtype=torch.double)
[n,d] = X.shape
y = y.unsqueeze(1)

# variables and corresponding dimensions.
var_in = {"w": [d,1], "b": [1,1]}

#generate a list of markers and another of colors 
markers = [ "," , "o" , "v" , "^" , "<", ">", "." ]
colors = ['r','g','b','c','m', 'y', 'k']
idx = 0 # index for plots
for maxfolding in folding_list:
    print('\n\n\n'+maxfolding + '  start!')
    idx+=1

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
                iter_lst.append(soln.iters)
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
    iter_arr = np.array(iter_lst)

    index_sort = np.argsort(F_arr)
    index_sort = index_sort[::-1]
    sorted_F = F_arr[index_sort]
    sorted_T = T_arr[index_sort]
    sorted_MF = MF_arr[index_sort]
    sorted_termination = term_arr[index_sort]
    sorted_acc = acc_arr[index_sort]
    sorted_tv = TV_arr[index_sort]
    sorted_mf_tv = MF_TV_arr[index_sort]
    sorted_iter = iter_arr[index_sort]

    print( "Time = {}".format(sorted_T) )
    print("F obj = {}".format(sorted_F))
    print("MF obj = {}".format(sorted_MF))
    print("termination code = {}".format(sorted_termination))
    print("train acc = {}".format(sorted_acc))
    print("total violation tvi + tve = {}".format(sorted_tv))
    print("MF total violation tvi + tve = {}".format(sorted_mf_tv))
    print("all termination code = {}".format(termination_lst_all))
    print('iterations = {}'.format(sorted_iter))

    arr_len = sorted_F.shape[0]
    plt.plot(np.arange(arr_len),sorted_acc,color = colors[idx], marker = markers[idx], linestyle = '-',label=maxfolding)
    # plt.plot(np.arange(arr_len),sorted_F,color = colors[idx], marker = markers[idx], linestyle = '-',label=maxfolding)
    # plt.plot(np.arange(arr_len),sorted_MF,'go-',label='sorted_pygranso_sol_MF')

plt.legend()        
plt.title(name_str + 'sorted train_acc' )
plt.xlabel('sorted random seeds')
plt.ylabel('training accuracy')
png_title =  "png/" + date_time + name_str
plt.savefig(os.path.join(my_path, png_title))

print("successful rate = {}".format(arr_len/total))

if write_to_log:
    # end writing
    sys.stdout.close()
