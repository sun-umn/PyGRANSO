#!/usr/bin/env python
# coding: utf-8

# # Sphere Manifold
# 
# This notebook contains examples of optimization on a sphere manifold.
# 
# Reference: https://www.manopt.org/manifold_documentation_sphere.html
# 

# ## Problem Description

# $$\min_{x}x^TAx,$$
# $$\text{s.t. }x^Tx=1,$$

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
torch.manual_seed(0)
n = 500
A = torch.randn((n,n)).to(device=device, dtype=torch.double)
A = .5*(A+A.T)


# ## Problem Definition
# 
# Spceify torch device, optimization variables, and corresponding objective and constrained function.
# 
# Note: please strictly follow the format of evalObjFunction and combinedFunction, which will be used in the NCVX main algortihm. *X_struct* is always required.

# In[3]:


# variables and corresponding dimensions.
var_in = {"x": [n,1]}


def comb_fn(X_struct):
    x = X_struct.x
    x.requires_grad_(True)
    
    # objective function
    f = x.T@A@x

    # inequality constraint, matrix form
    ci = None
    
    # equality constraint 
    ce = GeneralStruct()
    ce.c1 = x.T@x-1

    return [f,ci,ce]


# ## User Options
# Specify user-defined options for NCVX algorithm

# In[4]:


opts = Options()
opts.print_frequency = 10
opts.x0 = torch.randn((n,1)).to(device=device, dtype=torch.double)
opts.mu0 = 0.1 # increase penalty contribution
opts.opt_tol = 1e-6


# ## Main Algorithm

# In[5]:


start = time.time()
soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))

