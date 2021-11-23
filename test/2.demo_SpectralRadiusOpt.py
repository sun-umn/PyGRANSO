#!/usr/bin/env python
# coding: utf-8

# # Spectral Radius Optimization
# 
# This notebook contains examples of how to solve Spectral Radius Optimization problem.
# 
# Reference: Curtis, Frank E., Tim Mitchell, and Michael L. Overton. "A BFGS-SQP method for nonsmooth, nonconvex, constrained optimization and its evaluation using relative minimization profiles." Optimization Methods and Software 32.1 (2017): 148-181.

# ## Problem Description

# We have $M=A+BXC$,
# where the matirces $A\in R^{N,N},B\in R^{N,P}$ and $C\in R^{M,N}$ are given, $X\in R^{P,M}$ is the matrix form optimization variable.
# 
# We have the nonconvex, nonsmooth, and constrained optimization problem
# $$\min_{X}\rho_I(A+BXC),$$
# $$\text{s.t. }\rho_R(A+BXC)+\xi \leq 0,$$
# where $\xi$ is the stability margin, and $\rho_I$ and $\rho_R$ are the maximum imaginary and real part of singular values of $M$.

# ## Modules Importing
# Import all necessary modules and add NCVX src folder to system path.

# In[1]:


import time
import torch
import os,sys
## Adding NCVX directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/NCVX')
from ncvx import ncvx
from ncvxStruct import Options, Data, GeneralStruct 
import scipy.io
from torch import linalg as LA


# ## Data Generation 
# Specify torch device, and read the data from provided file

# In[2]:


device = torch.device('cuda')

# currentdir = os.path.dirname(os.path.realpath(__file__))
file = "/home/buyun/Documents/GitHub/NCVX/examples/data/spec_radius_opt_data.mat"
mat = scipy.io.loadmat(file)
mat_struct = mat['sys']
mat_struct = mat_struct[0,0]
A = torch.from_numpy(mat_struct['A']).to(device=device, dtype=torch.double)
B = torch.from_numpy(mat_struct['B']).to(device=device, dtype=torch.double)
C = torch.from_numpy(mat_struct['C']).to(device=device, dtype=torch.double)
p = B.shape[1]
m = C.shape[0]
stability_margin = 1


# ## Problem Definition 
# Spceify optimization variables and corresponding objective and constrained function.
# 
# Note: please strictly follow the format of evalObjFunction and combinedFunction, which will be used in the NCVX main algortihm. 

# In[3]:


# variables and corresponding dimensions.
var_in = {"X": [p,m] }

def comb_fn(X_struct):
    # user defined variable, matirx form. torch tensor
    X = X_struct.X
    X.requires_grad_(True)

    # objective function
    M           = A + B@X@C
    [D,_]       = LA.eig(M)
    f = torch.max(D.imag)

    # inequality constraint, matrix form
    ci = GeneralStruct()
    ci.c1 = torch.max(D.real) + stability_margin

    # equality constraint 
    ce = None
    
    return [f,ci,ce]


# ## User Options
# Specify user-defined options for NCVX algorithm

# In[4]:


opts = Options()
opts.maxit = 200
opts.x0 = torch.zeros(p*m,1).to(device=device, dtype=torch.double)
# print for every 10 iterations. default: 1
opts.print_frequency = 10


# ## Main Algorithm

# In[5]:


start = time.time()
soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))


# ## LBFGS 
# (Optional) LBFGS and feasibility related options

# In[6]:


opts = Options()
opts.maxit = 200
opts.x0 = torch.zeros(p*m,1).to(device=device, dtype=torch.double)
# print for every 10 iterations. default: 1
opts.print_frequency = 10

# Limited-memory mode is generally not recommended for nonsmooth
# problems, such as this one, but it can nonetheless enabled if
# desired/necessary.  opts.limited_mem_size == 0 is off, that is, 
# limited-memory mode is disabled.
# Note that this example has 200 variables.
opts.limited_mem_size = 40

start = time.time()
soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))


# In[7]:


# We can also tune NCVX to more aggressively favor satisfying
# feasibility over minimizing the objective.  Set feasibility_bias to
# true to adjust the following three steering parameters away from
# their default values.  For more details on these parameters, type
# import ncvxOptionsAdvanced
# help(ncvxOptionsAdvanced)
import numpy as np
opts = Options()
feasibility_bias = True
if feasibility_bias:
    opts.steering_ineq_margin = np.inf    # default is 1e-6
    opts.steering_c_viol = 0.9         # default is 0.1
    opts.steering_c_mu = 0.1           # default is 0.9


# In[8]:


opts.maxit = 200
opts.x0 = torch.zeros(p*m,1).to(device=device, dtype=torch.double)
# print for every 10 iterations. default: 1
opts.print_frequency = 10

start = time.time()
soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))


# In my testing, with default parameters, NCVX will first obtain a
# feasible solution at iter ~= 160 and will reduce the objective to
# 11.60 by the time it attains max iteration count of 200.
# 
# With feasibility_bias = True, in my testing, NCVX will obtain its
# first feasible solution earlier, at iter ~= 60, but it will ultimately
# have reduced the objective value less, only to 12.21, by the end of
# its 200 maximum allowed iterations.
