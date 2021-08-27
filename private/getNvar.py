import torch

def getNvar(var_dim_map):
    # calculate total number of scalar variables
    nvar = 0
    for dim in var_dim_map.values():
        nvar = nvar + dim[0] * dim[1]
    return nvar

def getNvarTorch(parameters):
   vec = torch.nn.utils.parameters_to_vector(parameters)
   return vec.shape[0]
