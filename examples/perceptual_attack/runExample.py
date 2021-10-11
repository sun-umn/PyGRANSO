import time
import numpy as np
import torch
import numpy.linalg as la
from scipy.stats import norm
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
sys.path.append('/home/buyun/Documents/GitHub/perceptual-advex')

from perceptual_advex.utilities import get_dataset_model
from perceptual_advex.perceptual_attacks import get_lpips_model
from perceptual_advex.distances import normalize_flatten_features


from pygranso import pygranso
from pygransoStruct import Options, Data
from private.getNvar import getNvarTorch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

def mainFun():
    # Please read the documentation on https://pygranso.readthedocs.io/en/latest/

    

    device = torch.device('cuda')
    # device = torch.device('cpu')
    dataset, model = get_dataset_model(
    dataset='cifar',
    arch='resnet50',
    checkpoint_fname='/home/buyun/Documents/GitHub/perceptual-advex/data/checkpoints/cifar_pgd_l2_1.pt',
)
    model = model.to(device=device, dtype=torch.double)
    # Create a validation set loader.
    batch_size = 1
    _, val_loader = dataset.make_loaders(1, batch_size, only_val=True, shuffle_val=False)

    


    # user defined options
    opts = Options()
    nvar = getNvarTorch(model.parameters())
    opts.QPsolver = 'osqp'
    opts.maxit = 100
    opts.opt_tol = 1e-6
    opts.fvalquit = 1e-6
    opts.print_level = 1
    opts.print_frequency = 1

    ################################################
    total_count = 100
    original_count = 0
    attack_count = 0
    # attack_type = 'L_2'
    # attack_type = 'L_inf'
    attack_type = 'Perceptual'
    total_time = 0
    total_iterations = 0

    for i in range(total_count):
        # Get a batch from the validation set.
        inputs, labels = next(iter(val_loader))
        inputs = inputs.to(device=device, dtype=torch.double)
        labels = labels.to(device=device)

        # variables and corresponding dimensions.
        var_in = {"x_tilde": list(inputs.shape)}

        opts.x0 = torch.reshape(inputs,(torch.numel(inputs),1))

        pred_labels = model(inputs).argmax(1)
        if pred_labels == labels:
            original_count += 1
        else:
            continue

        # print("Initial acc = {}".format(acc))

    

        data_in = Data()
        data_in.labels = labels.to(device=device)  # label/target [256]
        # input data [256,3,32,32]
        data_in.inputs = inputs.to(device=device, dtype=torch.double)
        data_in.model = model
        data_in.attack_type = attack_type

        if data_in.attack_type == 'Perceptual':
            lpips_model = get_lpips_model('alexnet_cifar', model)
            data_in.lpips_model = lpips_model.to(device=device, dtype=torch.double)

        #  main algorithm  
        start = time.time()
        soln = pygranso(var_dim_map = var_in, torch_device = device, user_data = data_in, user_opts = opts)
        end = time.time()
        print("Total Wall Time: {}s".format(end - start))

        total_time += end - start
        total_iterations += soln.fn_evals

        final_adv_input = torch.reshape(soln.final.x,inputs.shape)
        pred_labels2 = model(final_adv_input.to(device=device, dtype=torch.double)).argmax(1)

        if pred_labels2 == labels:
            attack_count += 1

        # print("adv acc final = {}".format(acc2))

        if data_in.attack_type == 'L_2':
            print("adv diff L2 = {}".format( ( torch.norm((inputs.to(device=device, dtype=torch.double) - final_adv_input).reshape(inputs.size()[0], -1)) )))
        elif data_in.attack_type == 'L_inf':
            print("adv diff Linf = {}".format( ( torch.norm((inputs.to(device=device, dtype=torch.double) - final_adv_input).reshape(inputs.size()[0], -1), float('inf') ) )))
        else:
            input_features = normalize_flatten_features( lpips_model.features(inputs)).detach()
            adv_features = lpips_model.features(final_adv_input)
            adv_features = normalize_flatten_features(adv_features)
            lpips_dists = torch.mean((adv_features - input_features).norm(dim=1))
            print("adv diff perceptual = {}".format(lpips_dists))
        

    print("\n\n\nNatural acc = {}".format(( original_count/total_count )))
    print("Success rate of attack = {}".format( (original_count-attack_count)/original_count ))
    print("Mean time = {}s, mean f_eval = {} iters".format(total_time/original_count,total_iterations/original_count))
    


if __name__ == "__main__":
   
    mainFun( )
