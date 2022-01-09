
import perceptual_advex



# In[2]:


import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso import pygranso
from pygransoStruct import pygransoStruct
from private.getNvar import getNvarTorch

sys.path.append('/home/buyun/Documents/GitHub/perceptual-advex')
from perceptual_advex.utilities import get_dataset_model
# from perceptual_advex.perceptual_attacks import get_lpips_model
from perceptual_advex.distances import normalize_flatten_features


########################################################################################
from typing import Optional, Union
import torch
import torchvision.models as torchvision_models
from torchvision.models.utils import load_state_dict_from_url
import math
from torch import nn
from torch.nn import functional as F
from typing_extensions import Literal

from perceptual_advex.distances import normalize_flatten_features, LPIPSDistance
from perceptual_advex.utilities import MarginLoss
from perceptual_advex.models import AlexNetFeatureModel, CifarAlexNet, FeatureModel
from perceptual_advex import utilities

_cached_alexnet: Optional[AlexNetFeatureModel] = None
_cached_alexnet_cifar: Optional[AlexNetFeatureModel] = None

def get_lpips_model(
    lpips_model_spec: Union[
        Literal['self', 'alexnet', 'alexnet_cifar'],
        FeatureModel,
    ],
    model: Optional[FeatureModel] = None,
) -> FeatureModel:
    global _cached_alexnet, _cached_alexnet_cifar

    lpips_model: FeatureModel

    if lpips_model_spec == 'self':
        if model is None:
            raise ValueError(
                'Specified "self" for LPIPS model but no model passed'
            )
        return model
    elif lpips_model_spec == 'alexnet':
        if _cached_alexnet is None:
            alexnet_model = torchvision_models.alexnet(pretrained=True)
            _cached_alexnet = AlexNetFeatureModel(alexnet_model)
        lpips_model = _cached_alexnet
        if torch.cuda.is_available():
            lpips_model.cuda()
    elif lpips_model_spec == 'alexnet_cifar':
        if _cached_alexnet_cifar is None:
            alexnet_model = CifarAlexNet()
            _cached_alexnet_cifar = AlexNetFeatureModel(alexnet_model)
        lpips_model = _cached_alexnet_cifar
        if torch.cuda.is_available():
            lpips_model.cuda()
        try:
            state = torch.load('data/checkpoints/alexnet_cifar.pt')
        except FileNotFoundError:
            state = load_state_dict_from_url(
                'https://perceptual-advex.s3.us-east-2.amazonaws.com/'
                'alexnet_cifar.pt',
                progress=True,
            )
        lpips_model.load_state_dict(state['model'])
    elif isinstance(lpips_model_spec, str):
        raise ValueError(f'Invalid LPIPS model "{lpips_model_spec}"')
    else:
        lpips_model = lpips_model_spec

    lpips_model.eval()
    return lpips_model



##########################################################################################




device = torch.device('cuda')

dataset, model = get_dataset_model(
dataset='cifar',
arch='resnet50',
checkpoint_fname='/home/buyun/Documents/GitHub/PyGRANSO/examples/data/checkpoints/cifar_pgd_l2_1.pt',
)
model = model.to(device=device, dtype=torch.double)
# Create a validation set loader.
batch_size = 1
_, val_loader = dataset.make_loaders(1, batch_size, only_val=True, shuffle_val=False)


# _, val_loader = dataset.make_loaders(1, batch_size, only_val=True, shuffle_val=True)

it = iter(val_loader)

# inputs, labels = next(iter(val_loader))
inputs, labels = next(it)
inputs, labels = next(it)
inputs, labels = next(it)
inputs, labels = next(it)
# inputs, labels = next(it)
# inputs, labels = next(it)
# inputs, labels = next(it)
# inputs, labels = next(it)
# inputs, labels = next(it)


# All the user-provided data (vector/matrix/tensor) must be in torch tensor format. 
# As PyTorch tensor is single precision by default, one must explicitly set `dtype=torch.double`.
# Also, please make sure the device of provided torch tensor is the same as opts.torch_device.
inputs = inputs.to(device=device, dtype=torch.double)
labels = labels.to(device=device)

# # # use different lpips model: alexnet_cifar; load pretrained alexnet_cifar model
lpips_model = get_lpips_model('alexnet_cifar', model).to(device=device, dtype=torch.double)

# # try to use alexnet for obj
model = get_lpips_model('alexnet_cifar', model).to(device=device, dtype=torch.double)

# use same lpips model: cifar_pgd_l2_1
# lpips_model = get_lpips_model('self', model).to(device=device, dtype=torch.double)

# attack_type = 'L_2'
# attack_type = 'L_inf
attack_type = 'Perceptual'


# ## Function Set-Up
# 
# Encode the optimization variables, and objective and constraint functions.
# 
# Note: please strictly follow the format of comb_fn, which will be used in the PyGRANSO main algortihm.

# In[12]:


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

def user_fn(X_struct,inputs,labels,lpips_model,model,attack_type):
    adv_inputs = X_struct.x_tilde
    
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

    # inequality constraint
    ci = pygransoStruct()
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

comb_fn = lambda X_struct : user_fn(X_struct,inputs,labels,lpips_model,model,attack_type)


# ## User Options
# Specify user-defined options for PyGRANSO 

# In[13]:


opts = pygransoStruct()
opts.torch_device = device
opts.maxit = 10000
# opts.opt_tol = 1e-3
opts.print_frequency = 10
opts.x0 = torch.reshape(inputs,(torch.numel(inputs),1))
# opts.init_step_size = 0.1


# ## Main Algorithm

# In[14]:


start = time.time()
soln = pygranso(var_spec = var_in,combined_fn = comb_fn,user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))

