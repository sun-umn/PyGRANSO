import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
import utils
import numpy as np
import os

#####################################

# debug_mode = True
debug_mode = False


attack_type = "Perceptual"
# attack_type = "L_inf"
# attack_type = "L_2"

box_constr = True

batch_size = 10
# batch_size = 50000

epsilon = 0.01 # attack bound
rng_seed = 0
opt_tol =  1e-4*np.sqrt(150528) # 1e-4*np.sqrt(torch.numel(inputs))
viol_ineq_tol = 5e-5
maxit = 50
print_frequency = 1
limited_mem_size = 10
mu0 = 1
maxclocktime = 30

device = torch.device('cuda')
# precision = torch.double
precision = torch.float
#####################################

torch.manual_seed(rng_seed)
base_model = utils.load_base_model(device,precision)
val_loader = utils.get_val_loader()

[my_path, log_name, date_time, name_str] = utils.get_name(batch_size,maxclocktime,attack_type,box_constr,rng_seed)
if not debug_mode:
    sys.stdout = open(os.path.join(my_path, log_name), 'w')

Loop_start = time.time()
result_dict = utils.result_dict_init() # initialize result dict
total_img = len(val_loader)
img_idx=0
for inputs, labels in val_loader:

    if img_idx > batch_size:
        break

    [inputs,labels] = utils.move_data_device(device,precision,inputs,labels)

    # variables and corresponding dimensions.
    var_in = {"x_tilde": list(inputs.shape)}
    comb_fn = lambda X_struct : utils.user_fn(X_struct, inputs, labels, lpips_model=base_model, model=base_model, attack_type=attack_type, eps=epsilon, box_constr=box_constr)
    opts = utils.get_opts(device,maxit,opt_tol,viol_ineq_tol,print_frequency,limited_mem_size,mu0,inputs,precision,maxclocktime)
    try:
        start = time.time()
        soln = pygranso(var_spec = var_in,combined_fn = comb_fn,user_opts = opts)
        end = time.time()
        print("image idx {} out of {} imgs; Wall Time: {}s".format(img_idx,total_img,end - start))
        result_dict = utils.store_result(soln,end,start,img_idx,result_dict)
    except Exception as e:
        print('skip pygranso')
    img_idx+=1

utils.sort_result(result_dict)
utils.print_result(result_dict,total_img)

Loop_end = time.time()
print("total Wall time of the whole loop = {}".format(Loop_end-Loop_start))

if debug_mode:
    utils.visualize_attack(soln,inputs)
else:
    data_name = utils.get_data_name(my_path, date_time, name_str)
    np.save(data_name,result_dict)
    # end writing
    sys.stdout.close()