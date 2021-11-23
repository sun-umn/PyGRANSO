#!/usr/bin/env python
# coding: utf-8

# # Nonlinear Feasibility Problem
# 
# This notebook contains examples of find a point that satisfies all the constraints in a problem, with no objective function to minimize.
# 
# Reference: https://www.mathworks.com/help/optim/ug/solve-feasibility-problem.html

# ## Problem Description

# $$(y+x^2)^2+0.1y^2\leq1$$
# $$y\leq\exp(-x)-3$$
# $$y\leq x-4$$ 
# Do any points $(x,y)$ satisfy all of the constraints?

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


# ## Problem Definition
# 
# Spceify torch device, optimization variables, and corresponding objective and constrained function.
# 
# Note: please strictly follow the format of evalObjFunction and combinedFunction, which will be used in the NCVX main algortihm. *X_struct* is always required.

# In[2]:


device = torch.device( 'cuda')

# variables and corresponding dimensions.
var_in = {"x": [1,1],"y": [1,1]}


def comb_fn(X_struct):
    x = X_struct.x
    y = X_struct.y
    x.requires_grad_(True)
    y.requires_grad_(True)
    # constant objective function
    f = 0*x+0*y

    # inequality constraint 
    ci = GeneralStruct()
    ci.c1 = (y+x**2)**2+0.1*y**2-1
    ci.c2 = y - torch.exp(-x) - 3
    ci.c3 = y-x+4
    
    # equality constraint 
    ce = None

    return [f,ci,ce]


# ## User Options
# Specify user-defined options for NCVX algorithm

# In[3]:


opts = Options()
opts.QPsolver = 'osqp' 
opts.print_frequency = 1
opts.x0 = 0 * torch.ones((2,1)).to(device=device, dtype=torch.double)


# ## Main Algorithm

# In[4]:


start = time.time()
soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))
print("NCVX finds a feaible point:(%f,%f)"%(soln.final.x[0],soln.final.x[1]) )

