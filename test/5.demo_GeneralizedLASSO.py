#!/usr/bin/env python
# coding: utf-8

# # Generalized LASSO
# 
# This notebook contains examples of how to solve total variation denoising problem.
# 
# Reference: Boyd, Stephen, Neal Parikh, and Eric Chu. Distributed optimization and statistical learning via the alternating direction method of multipliers. Now Publishers Inc, 2011.
# 

# ## Problem Description

# $$\min \frac{1}{2} ||Ax-b||_2^2+\lambda||Fx||_1,$$
# where $A$ is an indentity matrix and $F$ is a the difference matrix, in which case the above form reduces to
# $$\min_x \frac{1}{2}||x-b||_2^2+\lambda\sum_{i=1}^{n-1}|x_{i+1}-x_i|$$

# ## Modules Importing
# Import all necessary modules and add NCVX src folder to system path. NCVX src folder to system path.

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


device = torch.device( 'cpu')
n = 80
eta = 0.5 # parameter for penalty term
torch.manual_seed(1)
b = torch.rand(n,1)
pos_one = torch.ones(n-1)
neg_one = -torch.ones(n-1)
F = torch.zeros(n-1,n)
F[:,0:n-1] += torch.diag(neg_one,0) 
F[:,1:n] += torch.diag(pos_one,0)
F = F.to(device=device, dtype=torch.double)  # double precision requireed in torch operations 
b = b.to(device=device, dtype=torch.double)


# ## Problem Definition
# 
# Spceify torch device, optimization variables, and corresponding objective and constrained function.
# 
# Note: please strictly follow the format of evalObjFunction and combinedFunction, which will be used in the NCVX main algortihm. *X_struct* and *data_in* are always required.

# In[3]:


# variables and corresponding dimensions.
var_in = {"x": [n,1]}

def obj_eval_fn(X_struct):
    x = X_struct.x
    x.requires_grad_(True)
    
    # objective function
    f = (x-b).t() @ (x-b)  + eta * torch.norm( F@x, p = 1)
    return f

def comb_fn(X_struct):
    x = X_struct.x
    x.requires_grad_(True)
    
    # objective function
    f = (x-b).t() @ (x-b)  + eta * torch.norm( F@x, p = 1)
    
    # inequality constraint, matrix form
    ci = None
    # equality constraint 
    ce = None

    return [f,ci,ce]


# ## User Options
# Specify user-defined options for NCVX algorithm

# In[4]:


opts = Options()
opts.QPsolver = 'osqp' 
opts.x0 = torch.ones((n,1)).to(device=device, dtype=torch.double)
opts.print_level = 1
opts.print_frequency = 10


# ## Main Algorithm

# In[5]:


start = time.time()
soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))

