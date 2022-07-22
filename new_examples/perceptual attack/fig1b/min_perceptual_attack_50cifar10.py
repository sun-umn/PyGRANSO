import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO_timing')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch
from perceptual_advex.utilities import get_dataset_model
from perceptual_advex.perceptual_attacks import get_lpips_model
from perceptual_advex.distances import normalize_flatten_features
import gc


device = torch.device('cuda')

# dataset, model = get_dataset_model(
# dataset='imagenet100',
# dataset_path='/home/buyun/Documents/GitHub/PyGRANSO/examples/data',
# arch='resnet50',
# checkpoint_fname='/home/buyun/Documents/GitHub/PyGRANSO/examples/data/checkpoints/imagenet_checkpoint.pth',
# )

dataset, model = get_dataset_model(
dataset='cifar',
arch='resnet50',
checkpoint_fname='/home/buyun/Documents/GitHub/PyGRANSO/examples/data/checkpoints/cifar_pgd_l2_1.pt',
)

model = model.to(device=device, dtype=torch.double)
# Create a validation set loader.
batch_size = 1
_, val_loader = dataset.make_loaders(1, batch_size, only_val=True, shuffle_val=False)

inputs, labels = next(iter(val_loader))

# All the user-provided data (vector/matrix/tensor) must be in torch tensor format.
# As PyTorch tensor is single precision by default, one must explicitly set `dtype=torch.double`.
# Also, please make sure the device of provided torch tensor is the same as opts.torch_device.
inputs = inputs.to(device=device, dtype=torch.double)
labels = labels.to(device=device)

# externally-bounded attack: AlexNet for constraint while ResNet for objective
lpips_model = get_lpips_model('alexnet_cifar', model).to(device=device, dtype=torch.double)

# Don't reccoment use in the current version. self-bounded attack: AlexNet for both constraint and objective
# model = get_lpips_model('alexnet_cifar', model).to(device=device, dtype=torch.double)



# variables and corresponding dimensions.
var_in = {"delta": list(inputs.shape)}

def MarginLoss(logits,labels):
    correct_logits = torch.gather(logits, 1, labels.view(-1, 1))
    max_2_logits, argmax_2_logits = torch.topk(logits, 2, dim=1)

    top_max, second_max = max_2_logits.chunk(2, dim=1)
    top_argmax, _ = argmax_2_logits.chunk(2, dim=1)
    labels_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
    labels_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
    max_incorrect_logits = labels_eq_max * second_max + labels_ne_max * top_max
    loss = -(max_incorrect_logits - correct_logits).clamp(max=2).squeeze(1)
    return loss

def user_fn(X_struct,inputs,labels,lpips_model,model):

    delta = X_struct.delta
    # alpha = X_struct.alpha
    # alpha = alpha.to(dtype=torch.float)

    # objective
    # if attack_type == 'L2':
    # f = torch.linalg.vector_norm(delta,2)
    # elif attack_type == 'L1':
    f = torch.linalg.vector_norm(delta,1)/(inputs.numel())**0.5

    # inequality constraint
    ci = pygransoStruct()
    # ci.c1 = torch.sum(torch.clamp(-alpha, min=0)**2)**0.5 # l2

    adv_inputs = inputs + delta
    logits_outputs = model(adv_inputs)
    k = torch.numel(logits_outputs)
    fc = logits_outputs[:,labels] # true class output
    fl = torch.hstack((logits_outputs[:,0:labels],logits_outputs[:,labels+1:k])) # remove the true class output
    ci.c1 = fc - torch.max(fl)
    
    # Input Box Constraint
    ci.c2 = torch.hstack(
        (adv_inputs.reshape(inputs.numel())-1,
        -adv_inputs.reshape(inputs.numel()))
    )
    # ci.c2 = torch.sum(torch.clamp(ci.c2, min=0)**2)**0.5 # l2
    # equality constraint
    ce = None

    return [f,ci,ce]


comb_fn = lambda X_struct : user_fn(X_struct,inputs,labels,lpips_model,model)

opts = pygransoStruct()
opts.torch_device = device
opts.maxit = 1000
opts.opt_tol = 1e-6
opts.print_frequency = 1
opts.limited_mem_size = 10
opts.x0 = torch.reshape(inputs,(torch.numel(inputs),1))

start = time.time()
soln = pygranso(var_spec = var_in,combined_fn = comb_fn,user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))