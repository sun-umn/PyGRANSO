#!/usr/bin/env python
# coding: utf-8

# # Dictionary Learning
# 
# This notebook contains examples of how to solve orthogonal dictionary learning problem.
# 
# Reference: Bai, Yu, Qijia Jiang, and Ju Sun. "Subgradient descent learns orthogonal dictionaries." arXiv preprint arXiv:1810.10702 (2018).

# ## Problem Description

# given data $\{y_i \}_{i \in[m]}$ generated as $y_i = A x_i$, where $A \in R^{n \times n}$ is a fixed unknown orthogonal matrix and each $x_i \in R^n$ is an iid Bernoulli-Gaussian random vector with parameter $\theta \in (0,1)$, recover $A$. 
# 
# Write $Y \doteq [y_1,...,y_m]$ and $X \doteq [vx_1,...,x_m]$. Nonconvex due to constraint, nonsmooth due to objective:
# 
# $$\min_{q \in R^n} f(q) \doteq \frac{1}{m} ||q^T Y||_{1} = \frac{1}{m} \sum_{i=1}^m |q^T y_i|,$$
# $$\text{s.t.} ||q||_2 = 1,$$
# 
# 
# Based on above statistical model, $q^T Y = q^T A X$ has the highest sparsity when $q$ is a column of $A$ (up to sign) so that $q^T A$ is 1-sparse. 

# ## Modules Importing
# Import all necessary modules and add NCVX src folder to system path.

# In[1]:


import time
import numpy as np
import torch
import numpy.linalg as la
from scipy.stats import norm
import sys
## Adding NCVX directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/NCVX')
from ncvx import ncvx
from ncvxStruct import Options, GeneralStruct 


# ## Data Generation 
# Specify torch device, and generate data
# 
# Use gpu for this problem. If no cuda device available, please set *device = torch.device('cpu')*

# In[2]:


device = torch.device('cuda')
n = 30

np.random.seed(1)
m = 10*n**2   # sample complexity
theta = 0.3   # sparsity level
Y = norm.ppf(np.random.rand(n,m)) * (norm.ppf(np.random.rand(n,m)) <= theta)  # Bernoulli-Gaussian model
Y = torch.from_numpy(Y).to(device=device, dtype=torch.double)


# ## Problem Definition
# 
# Spceify torch device, optimization variables, and corresponding objective and constrained function.
# 
# Note: please strictly follow the format of evalObjFunction and combinedFunction, which will be used in the NCVX main algortihm. *X_struct* and *data_in* are always required.

# In[3]:


# variables and corresponding dimensions.
var_in = {"q": [n,1]}


def comb_fn(X_struct):
    q = X_struct.q
    q.requires_grad_(True)
    
    # objective function
    qtY = q.T @ Y
    f = 1/m * torch.norm(qtY, p = 1)

    # inequality constraint, matrix form
    ci = None

    # equality constraint 
    ce = GeneralStruct()
    ce.c1 = q.T @ q - 1

    return [f,ci,ce]


# ## User Options
# Specify user-defined options for NCVX algorithm

# In[4]:


opts = Options()
opts.QPsolver = 'osqp' 
opts.maxit = 500
np.random.seed(1)
x0 = norm.ppf(np.random.rand(n,1))
x0 /= la.norm(x0,2)
opts.x0 = torch.from_numpy(x0).to(device=device, dtype=torch.double)

opts.print_frequency = 10


# ## Main Algorithm

# In[5]:


start = time.time()
soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))
print(max(abs(soln.final.x))) # should be close to 1


# ## Various Options
# 
# **(Optional)** Set fvalquit. Quit if objective function drops below this value at a feasible 
# iterate (that is, satisfying feasibility tolerances 
# opts.viol_ineq_tol and opts.viol_eq_tol)
# 
# In the example below, we get termination code 2 since the target objective reached at point feasible to tolerances

# In[6]:


opts = Options()
opts.QPsolver = 'osqp' 
opts.maxit = 500
np.random.seed(1)
x0 = norm.ppf(np.random.rand(n,1))
x0 /= la.norm(x0,2)
opts.x0 = torch.from_numpy(x0).to(device=device, dtype=torch.double)
opts.print_frequency = 10
opts.print_ascii = True


opts.fvalquit = 0.4963

soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
print(max(abs(soln.final.x))) # should be close to 1


# Set opt_tol. Tolerance for reaching (approximate) optimality/stationarity.
# See opts.ngrad, opts.evaldist, and the description of NCVX's 
# output argument soln, specifically the subsubfield .dnorm for more
# information.
# 
# In the result below, NCVX terminated when stationarity is below 1e-4

# In[7]:


opts = Options()
opts.QPsolver = 'osqp' 
opts.maxit = 500
np.random.seed(1)
x0 = norm.ppf(np.random.rand(n,1))
x0 /= la.norm(x0,2)
opts.x0 = torch.from_numpy(x0).to(device=device, dtype=torch.double)
opts.print_frequency = 10
opts.print_ascii = True

opts.opt_tol = 1e-4 # default 1e-8

soln = ncvx(combinedFunction = comb_fn ,var_dim_map = var_in, torch_device = device, user_opts = opts)
print(max(abs(soln.final.x))) # should be close to 1


# There are multiple other settings. Please uncomment to try them. Detailed description can be found by typing
# 
# import ncvxOptionsAdvanced
# 
# help(ncvxOptionsAdvanced)

# In[8]:


opts = Options()
opts.QPsolver = 'osqp' 
opts.maxit = 500
np.random.seed(1)
x0 = norm.ppf(np.random.rand(n,1))
x0 /= la.norm(x0,2)
opts.x0 = torch.from_numpy(x0).to(device=device, dtype=torch.double)
opts.print_frequency = 10

# Please uncomment to try different settings

# Tolerance for determining when the relative decrease in the penalty
# function is sufficiently small.  NCVX will terminate if when 
# the relative decrease in the penalty function is at or below this
# tolerance and the current iterate is feasible to tolerances.
# Generally, we don't recommend using this feature since small steps
# are not necessarily indicative of being near a stationary point,
# particularly for nonsmooth problems.

# Termination Code 1
# opts.rel_tol = 1e-2 # default 0

# Tolerance for how small of a step the line search will attempt
# before terminating.

# Termination Code 6 or 7
# opts.step_tol = 1e-6 # default 1e-12
# opts.step_tol = 1e-3

# Acceptable total violation tolerance of the equality constraints.
# opts.viol_eq_tol = 1e-12# default 1e-6, make it smaller will make current point harder to be considered as feasible

# Quit if the elapsed clock time in seconds exceeds this. unit: second
# opts.maxclocktime = 1.

# Number of characters wide to print values for the penalty function,
# the objective function, and the total violations of the inequality 
# and equality constraints. 
# opts.print_width = 9

# NCVX's uses orange
# printing to highlight pertinent information.  However, the user
# is the given option to disable it, since support cannot be
# guaranteed (since it is an undocumented feature).
# opts.print_use_orange = False

# opts.init_step_size = 1e-2
# opts.searching_direction_rescaling = True

soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
print(max(abs(soln.final.x))) # should be close to 1

