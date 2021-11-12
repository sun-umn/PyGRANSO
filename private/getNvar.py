import torch
import numpy as np

def getNvar(var_dim_map):
    """
    getNvar calculates the total number of scalar variables
    """
    nvar = 0
    for dim in var_dim_map.values():
        # nvar = nvar + dim[0] * dim[1]
        nvar = nvar + np.prod(dim)
    return int(nvar)

def getNvarTorch(parameters):
   vec = torch.nn.utils.parameters_to_vector(parameters)
   return vec.shape[0]
