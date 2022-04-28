import torch
import utils
import sys
import os
import numpy as np
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
import time
from matplotlib import pyplot as plt 


#generate a list of markers and another of colors 
markers = [ "," , "o" , "v" , "^" , "<", ">", "." ]
colors = ['r','g','b','c','m', 'y', 'k']

###############################################
debug_mode = False

n = 10 # V: n*d
d = 5 # copnst: d*d

# folding_list = ['l2','l1','linf']
# folding_list = ['l2','l1']
folding_list = ['l2','l1','linf','unfolding']
# folding_list = ['l2','unfolding']

K = 3 # K number of starting points. random generated initial guesses
N = 2 # number of different data matrix (with the same size)

opt_tol = 1e-6
maxit = 50000
maxclocktime = 30
# QPsolver = "gurobi"
QPsolver = "osqp"
threshold = 0.99

# square_flag = True
square_flag = False

mu0 = 0.1
device = torch.device('cuda')


###############################################

[my_path, log_name, date_time, name_str] = utils.get_name(square_flag,folding_list,n,d,K,maxclocktime)

if not debug_mode:
    sys.stdout = open(os.path.join(my_path, log_name), 'w')

###################################################

for rng_seed in range(N):
    [A, U, ana_sol] = utils.data_init(rng_seed, n, d, device)
    folding_idx = 0 # index for plots
    for maxfolding in folding_list:
        print('\n\n\n'+ maxfolding + '  start!')
        folding_idx+=1
        comb_fn = lambda X_struct : utils.user_fn(X_struct,A,d,device,maxfolding,square_flag)
        result_dict = utils.result_dict_init()
        start_loop = time.time()
        for i in range(K):
            print("the {}th test out of K = {} initial guesses.\n rng seed {} out of N = {} ".format(i+1,K, rng_seed,N))
            try:
                # call pygranso
                start = time.time()
                var_in = {"V": [n,d]}
                opts = utils.opts_init(device,maxit,opt_tol,maxclocktime,QPsolver,mu0,ana_sol,threshold,n,d)
                soln = pygranso(var_spec = var_in,combined_fn = comb_fn,user_opts = opts)
                end = time.time()
                result_dict = utils.store_result(soln,end,start,n,d,i,result_dict,U)
            except Exception as e:
                print('skip pygranso')
        end_loop = time.time()
        print("K Loop Wall Time for this folding: {}s".format(end_loop - start_loop))
        utils.sort_result(result_dict)
        arr_len = utils.print_result(result_dict,K)
        # plot
        plt.plot(np.arange(arr_len),result_dict['F'],color = colors[folding_idx], marker = markers[folding_idx], linestyle = '-',label=maxfolding)

    plt.plot(np.arange(arr_len),np.array(arr_len*[ana_sol]),color = colors[folding_idx+1], linestyle = '-',label='analytical sol')
    plt.legend()
    plt.title(name_str)
    plt.xlabel('sorted sample index')
    plt.ylabel('obj val')
    

    if not debug_mode:
        [data_name,png_title] = utils.add_path(my_path,rng_seed, date_time, name_str)
        np.save(data_name,result_dict)
        plt.savefig(os.path.join(my_path, png_title))
        plt.clf()
    else:
        plt.show()

if not debug_mode:
    # end writing
    sys.stdout.close()


