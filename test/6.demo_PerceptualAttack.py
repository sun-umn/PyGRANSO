#!/usr/bin/env python
# coding: utf-8

# # Perceptual Attack
# 
# This notebook contains examples of how to solve perceptual attack problems 
# 
# Reference: Laidlaw, Cassidy, Sahil Singla, and Soheil Feizi. "Perceptual adversarial robustness: Defense against unseen threat models." arXiv preprint arXiv:2006.12655 (2020).

# ## Problem Description

# Given classifier $f$ which could map input image $x \in X$ to labels $y = f(x) \in Y$0. The goal of neural perceptual attack is to find an input $\tilde{x}$ that is perceptually similar to the original input $x$ but can fool the classifier $f$, which can be formulated as:
# 
# $$\max_{\tilde{x}} L (f(\tilde{x}),y),$$
# $$\text{s.t.}\;\; d(x,\tilde{x}) = ||\phi(x) - \phi (\tilde{x}) ||_{2} \leq \epsilon$$
# Here $$L (f({x}),y) = \max_{i\neq y} (z_i(x) - z_y(x) ),$$
# where $z_i(x)$ is the $i$-th logit output of $f(x)$, and $\phi(\cdot)$ is a function that could map the input $x$ to  normalized, flattened activations

# ## Modules Importing
# Import all necessary modules and add NCVX src folder to system path. NCVX src folder to system path.
# 
# NOTE: perceptual advex source code (https://github.com/cassidylaidlaw/perceptual-advex.git) are required to calculate the lpips distance 

# In[1]:


import time
import torch
import sys
## Adding NCVX directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/NCVX')
from ncvx import ncvx
from ncvxStruct import Options, GeneralStruct 
from private.getNvar import getNvarTorch

sys.path.append('/home/buyun/Documents/GitHub/perceptual-advex')
from perceptual_advex.utilities import get_dataset_model
from perceptual_advex.perceptual_attacks import get_lpips_model
from perceptual_advex.distances import normalize_flatten_features


# ## Data Generation 
# 
# Specify torch device, neural network architecture, and generate data.
# 
# NOTE: please specify path for downloading data

# In[2]:


device = torch.device('cuda')

dataset, model = get_dataset_model(
dataset='cifar',
arch='resnet50',
checkpoint_fname='/home/buyun/Documents/GitHub/perceptual-advex/data/checkpoints/cifar_pgd_l2_1.pt',
)
model = model.to(device=device, dtype=torch.double)
# Create a validation set loader.
batch_size = 1
_, val_loader = dataset.make_loaders(1, batch_size, only_val=True, shuffle_val=False)

inputs, labels = next(iter(val_loader))
inputs = inputs.to(device=device, dtype=torch.double)
labels = labels.to(device=device)
lpips_model = get_lpips_model('alexnet_cifar', model).to(device=device, dtype=torch.double)

# attack_type = 'L_2'
# attack_type = 'L_inf'
attack_type = 'Perceptual'


# ## Problem Definition
# 
# Spceify torch device, optimization variables, and corresponding objective and constrained function.
# 
# Note: please strictly follow the format of evalObjFunction and combinedFunction, which will be used in the NCVX main algortihm. *X_struct* and *data_in* are always required.

# In[3]:


# variables and corresponding dimensions.
var_in = {"x_tilde": list(inputs.shape)}

def MarginLoss(logits,labels):
    correct_logits = torch.gather(logits, 1, labels.view(-1, 1))
    max_2_logits, argmax_2_logits = torch.topk(logits, 2, dim=1)
    top_max, second_max = max_2_logits.chunk(2, dim=1)
    top_argmax, _ = argmax_2_logits.chunk(2, dim=1)
    labels_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
    labels_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
    max_incorrect_logits = labels_eq_max * second_max + labels_ne_max * top_max
    loss = -(max_incorrect_logits - correct_logits).clamp(max=1).squeeze().sum()
    return loss

def comb_fn(X_struct):
    adv_inputs = X_struct.x_tilde
    adv_inputs.requires_grad_(True)
    
    # objective function
    # 8/255 for L_inf, 1 for L_2, 0.5 for PPGD/LPA
    if attack_type == 'L_2':
        epsilon = 1
    elif attack_type == 'L_inf':
        epsilon = 8/255
    else:
        epsilon = 0.5

    logits_outputs = model(adv_inputs)

    f = MarginLoss(logits_outputs,labels)

    # inequality constraint, matrix form
    # ci = None
    ci = GeneralStruct()
    if attack_type == 'L_2':
        ci.c1 = torch.norm((inputs - adv_inputs).reshape(inputs.size()[0], -1)) - epsilon
    elif attack_type == 'L_inf':
        ci.c1 = torch.norm((inputs - adv_inputs).reshape(inputs.size()[0], -1), float('inf')) - epsilon
    else:
        input_features = normalize_flatten_features( lpips_model.features(inputs)).detach()
        adv_features = lpips_model.features(adv_inputs)
        adv_features = normalize_flatten_features(adv_features)
        lpips_dists = (adv_features - input_features).norm(dim=1)
        ci.c1 = lpips_dists - epsilon
    
    # equality constraint 
    ce = None

    return [f,ci,ce]


# ## User Options
# Specify user-defined options for NCVX algorithm

# In[4]:


opts = Options()
opts.QPsolver = 'osqp'
opts.maxit = 100
opts.opt_tol = 1e-6
opts.fvalquit = 1e-6
opts.print_level = 1
opts.print_frequency = 10
opts.x0 = torch.reshape(inputs,(torch.numel(inputs),1))


# ## Main Algorithm

# In[5]:


start = time.time()
soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))


# ## Batch Attacks
# 
# Apply attack on multiple images by repeating above steps and calculate the successful rate

# In[6]:


total_count = 10
total_diff = 0
original_count = 0
attack_count = 0
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
    # suppress output
    opts.print_level = 0

    pred_labels = model(inputs).argmax(1)
    if pred_labels == labels:
        original_count += 1
    else:
        continue
    
    start = time.time()
    soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
    end = time.time()
    print("i = %d"%i)
    
    total_time += end - start
    total_iterations += soln.fn_evals

    final_adv_input = torch.reshape(soln.final.x,inputs.shape)
    pred_labels2 = model(final_adv_input.to(device=device, dtype=torch.double)).argmax(1)

    if pred_labels2 == labels:
        attack_count += 1
        
    if attack_type == 'L_2':
            diff = torch.norm((inputs.to(device=device, dtype=torch.double) - final_adv_input).reshape(inputs.size()[0], -1))
    elif attack_type == 'L_inf':
        diff = ( torch.norm((inputs.to(device=device, dtype=torch.double) - final_adv_input).reshape(inputs.size()[0], -1), float('inf') ) )
    else:
        input_features = normalize_flatten_features( lpips_model.features(inputs)).detach()
        adv_features = lpips_model.features(final_adv_input)
        adv_features = normalize_flatten_features(adv_features)
        lpips_dists = torch.mean((adv_features - input_features).norm(dim=1))
        diff = lpips_dists

    total_diff += diff

print("\n\n\nModel train acc on the original image = {}".format(( original_count/total_count )))
print("Success rate of attack = {}".format( (original_count-attack_count)/original_count ))
print("Average distance between attacked image and original image = {}".format(total_diff/original_count))
print("Average run time of NCVX = {}s, mean f_eval = {} iters".format(total_time/original_count,total_iterations/original_count))

