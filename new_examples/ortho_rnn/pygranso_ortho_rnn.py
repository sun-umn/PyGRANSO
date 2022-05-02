import utils
import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
import numpy as np
import os

maxfolding = 'l2'
# unconstrained = True
unconstrained = False

maxclocktime = 240
maxit = 50
row_by_row = False

save_plot = True

N = 16 # total random seeds number

restart_time = 30

device = torch.device('cuda')

if not row_by_row:
    sequence_length = 28*28
    input_size = 1
else:
    sequence_length = 28
    input_size = 28

hidden_size = 30
# hidden_size = 50

num_layers = 1
num_classes = 10

train_size=6000 # train acc 100; test acc 19%
test_size=1000

# train_size=60000 # train acc 100; test acc 19%
# test_size=10000

double_precision = torch.double
# double_precision = torch.float

# rng_seed = 5
# rng_seed = 6

# limited_mem_size = 10
limited_mem_size = 0


debug_mode = False


###############################################
title = utils.get_title(row_by_row,unconstrained,maxfolding,train_size,test_size,0,hidden_size)
[my_path, log_name] = utils.get_logname(title)

if not debug_mode:
    sys.stdout = open(os.path.join(my_path, log_name), 'w')

for rng_seed in range(N):

    title = utils.get_title(row_by_row,unconstrained,maxfolding,train_size,test_size,rng_seed,hidden_size)

    print('{} starts!'.format(title))

    model = utils.get_model(rng_seed,input_size, hidden_size, num_layers, num_classes,device,double_precision,sequence_length)

    [X_train, y_train, X_test, y_test] = utils.get_data(sequence_length, input_size, device, double_precision,train_size,test_size)

    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    comb_fn = lambda model : utils.user_fn(model,X_train,y_train,hidden_size,device,double_precision,maxfolding,unconstrained)

    opts = utils.get_opts(device,model,maxit,maxclocktime,double_precision,limited_mem_size)

    train_acc_list = np.array([])
    test_acc_list = np.array([])

    train_acc_list = utils.get_model_acc(model,X_train,y_train,True,train_acc_list)
    test_acc_list = utils.get_model_acc(model,X_test,y_test,False,test_acc_list)

    start = time.time()
    soln = pygranso(var_spec= model, combined_fn = comb_fn, user_opts = opts)
    # print acc
    torch.nn.utils.vector_to_parameters(soln.final.x, model.parameters())
    train_acc_list = utils.get_model_acc(model,X_train,y_train,True,train_acc_list)
    test_acc_list = utils.get_model_acc(model,X_test,y_test,False,test_acc_list)

    for i in range(restart_time):
        print('{}th restart'.format(i))
        opts = utils.get_restart_opts(device,model,maxit,maxclocktime,double_precision,soln,limited_mem_size,unconstrained)
        

        print("the {}th restart out of K = {} restart times.\n rng seed {} out of N = {}.  ".format(i+1,restart_time, rng_seed,N))
        try:
            # call pygranso
            soln = pygranso(var_spec= model, combined_fn = comb_fn, user_opts = opts)
        except Exception as e:
            print('skip pygranso')
        
        # print acc
        torch.nn.utils.vector_to_parameters(soln.final.x, model.parameters())
        train_acc_list = utils.get_model_acc(model,X_train,y_train,True,train_acc_list)
        test_acc_list = utils.get_model_acc(model,X_test,y_test,False,test_acc_list)

    end = time.time()
    print("Total Wall Time: {}s".format(end - start))


    utils.make_plot(train_acc_list,test_acc_list,restart_time,maxit,title,save_plot)

if not debug_mode:
    # end writing
    sys.stdout.close()
