import time
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

def get_name(square_flag,folding_list,n,d,total,maxclocktime):
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
    name_str = "_n{}_d{}_{}{}_total{}_maxtime{}".format(n,d,square_str,maxfolding,total,maxclocktime)
    log_name = "log/" + date_time + name_str + '.txt'

    print( name_str + "start\n\n")
    return [my_path, log_name, date_time, name_str]


def data_init(rng_seed, n, d, device):
    # fix random seeds
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    # data initialization
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
    ana_sol = -torch.trace(U.T@A@U).item()
    return [A, U, ana_sol]

def user_fn(X_struct,A,d,device,maxfolding,square_flag):
    V = X_struct.V
    # objective function
    f = -torch.trace(V.T@A@V)
    # inequality constraint, matrix form
    ci = None
    # equality constraint
    ce = pygransoStruct()
    constr_vec = (V.T@V - torch.eye(d).to(device=device, dtype=torch.double)).reshape(d**2,1)
    if maxfolding == 'l1':
        ce.c1 = torch.sum(torch.abs(constr_vec))
    elif maxfolding == 'l2':
        ce.c1 = torch.sum(constr_vec**2)**0.5
    elif maxfolding == 'linf':
        ce.c1 = torch.amax(torch.abs(constr_vec))
    elif maxfolding == 'unfolding':
        ce.c1 = V.T@V - torch.eye(d).to(device=device, dtype=torch.double)
    else:
        print("Please specficy you maxfolding type!")
        exit()
    if square_flag:
        ce.c1 = ce.c1**2
    return [f,ci,ce]

def opts_init(device,maxit,opt_tol,maxclocktime,QPsolver,mu0,ana_sol,threshold,n,d):
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
    opts.fvalquit = ana_sol*threshold
    opts.x0 =  torch.randn((n*d,1)).to(device=device, dtype=torch.double)
    opts.x0 = opts.x0/norm(opts.x0)
    return opts

def result_dict_init():
    result_dict = {
        'time':np.array([]),
        'iter': np.array([]),
        'F': np.array([]),
        'MF': np.array([]),
        'term_code_pass': np.array([]),
        'tv': np.array([]),
        'MF_tv': np.array([]),
        'term_code_fail': [],
        'E': np.array([]), # error
        'index_sort': np.array([])
        }
    return result_dict

def store_result(soln,end,start,n,d,i,result_dict,U):
    print("Total Wall Time: {}s".format(end - start))
    if soln.termination_code != 12 and soln.termination_code != 8:
        # mean error
        V = torch.reshape(soln.final.x,(n,d))
        E = norm(V-U)/norm(U)
        result_dict['E'] = np.append(result_dict['E'],E.item())
        result_dict['time'] = np.append(result_dict['time'],end-start)
        result_dict['F'] = np.append(result_dict['F'],soln.final.f)
        result_dict['MF'] = np.append(result_dict['MF'],soln.most_feasible.f)
        result_dict['term_code_pass'] = np.append(result_dict['term_code_pass'],soln.termination_code)
        result_dict['tv'] = np.append(result_dict['tv'],soln.final.tv) #total violation at x (vi + ve)
        result_dict['MF_tv'] = np.append(result_dict['MF_tv'],soln.most_feasible.tv)
        result_dict['iter'] = np.append(result_dict['iter'],soln.iters)
    else:
        result_dict['term_code_fail'].append("i = {}, code = {}\n ".format(i,soln.termination_code) )

    return result_dict

def sort_result(result_dict):
    index_sort = np.argsort(result_dict['F'])
    index_sort = index_sort[::-1]
    result_dict['F'] = result_dict['F'][index_sort]
    result_dict['E'] = result_dict['E'][index_sort]
    result_dict['time'] = result_dict['time'][index_sort]
    result_dict['MF'] = result_dict['MF'][index_sort]
    result_dict['term_code_pass'] = result_dict['term_code_pass'][index_sort]
    result_dict['tv'] = result_dict['tv'][index_sort]
    result_dict['MF_tv'] = result_dict['MF_tv'][index_sort]
    result_dict['iter'] = result_dict['iter'][index_sort]
    result_dict['index_sort'] = index_sort

def print_result(result_dict,total):
    print("Time = {}".format(result_dict['time']) )
    print("F obj = {}".format(result_dict['F']))
    print("MF obj = {}".format(result_dict['MF']))
    print("termination code = {}".format(result_dict['term_code_pass']))
    print("total violation tvi + tve = {}".format(result_dict['tv']))
    print("MF total violation tvi + tve = {}".format(result_dict['MF_tv']))
    print('iterations = {}'.format(result_dict['iter']))
    print("Error = {}".format(result_dict['E']))
    print("index sort = {}".format(result_dict['index_sort']))
    print("failed code: {}".format(result_dict['term_code_fail']))

    arr_len = result_dict['F'].shape[0]
    print("successful rate = {}".format(arr_len/total))
    return arr_len

def add_path(my_path,rng_seed, date_time, name_str):
    png_title =  "png/sorted_F_" +  date_time + '_seed_{}'.format(rng_seed) + name_str
    data_name =  'data/' + date_time + '_seed_{}'.format(rng_seed) + name_str +'.npy'
    data_name = os.path.join(my_path, data_name)
    png_title = os.path.join(my_path, png_title)
    return [data_name,png_title]