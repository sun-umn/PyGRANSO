#!/usr/bin/env python
# coding: utf-8

# # Robust PCA
# 
# This notebook contains examples of how to solve Robust PCA problem.
# 
# Reference: Yi, Xinyang, et al. "Fast algorithms for robust PCA via gradient descent." Advances in neural information processing systems. 2016.
# 

# ## Problem Description

# $$\min_{M,S}||M||_{\text{nuc}}+\lambda||S||_1,$$
# $$\text{s.t. }Y=M+S,$$
# where $M,S\in R^{d_1,d_2}$ are matrix form optimization variables, $Y\in R^{d_1,d_2}$ is a given matrix, and $||\cdot||_{\text{nuc}}$ denotes the nuclear norm.

# ## Modules Importing
# Import all necessary modules and add NCVX src folder to system path.

# In[1]:


import time
import torch
import sys
## Adding NCVX directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/NCVX')
from ncvx import ncvx
from ncvxStruct import Options, GeneralStruct 


# ## Data Generation 
# Specify torch device, and generate data

# In[2]:


device = torch.device( 'cuda')
d1 = 5
d2 = 10
torch.manual_seed(1)
eta = .5
Y = torch.randn(d1,d2).to(device=device, dtype=torch.double)


# ## Problem Definition
# 
# Spceify torch device, optimization variables, and corresponding objective and constrained function.
# 
# Note: please strictly follow the format of evalObjFunction and combinedFunction, which will be used in the NCVX main algortihm. *X_struct* is always required.

# In[3]:


# variables and corresponding dimensions.
var_in = {"M": [d1,d2],"S": [d1,d2]}


def comb_fn(X_struct):
    M = X_struct.M
    S = X_struct.S
    M.requires_grad_(True)
    S.requires_grad_(True)
    
    # objective function
    f = torch.norm(M, p = 'nuc') + eta * torch.norm(S, p = 1)

    # inequality constraint, matrix form
    ci = None
    
    # equality constraint 
    ce = GeneralStruct()
    ce.c1 = M + S - Y

    return [f,ci,ce]


# ## User Options
# Specify user-defined options for NCVX algorithm

# In[4]:


opts = Options()
opts.QPsolver = 'osqp' 
opts.print_frequency = 10
opts.x0 = .2 * torch.ones((2*d1*d2,1)).to(device=device, dtype=torch.double)


# ## Main Algorithm

# In[5]:


start = time.time()
soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))

