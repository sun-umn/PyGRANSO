#!/usr/bin/env python
# coding: utf-8

# # Rosenbrock
# 
# This notebook contains examples of solving optimization problem with 2-variable nonsmooth Rosenbrock objective function, which subject to simple bound constraints.
# 
# Reference: Curtis, Frank E., Tim Mitchell, and Michael L. Overton. "A BFGS-SQP method for nonsmooth, nonconvex, constrained optimization and its evaluation using relative minimization profiles." Optimization Methods and Software 32.1 (2017): 148-181.

# ## Problem Description

# $$\min_{x_1,x_2} w|x_1^2-x_2|+(1-x_1)^2,$$
# $$\text{s.t. }c_1(x_1,x_2) = \sqrt{2}x_1-1 \leq 0, c_(x_1,x_2)=2x_2-1\leq0,$$
# 
# where $w$ is a constant (e.g., $w=8$)

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
# Note: please strictly follow the format of evalObjFunction and combinedFunction, which will be used in the NCVX main algortihm.

# In[2]:


device = torch.device('cpu')
# variables and corresponding dimensions.
var_in = {"x1": [1,1], "x2": [1,1]}

def comb_fn(X_struct):
    x1 = X_struct.x1
    x2 = X_struct.x2
    # enable autodifferentiation
    x1.requires_grad_(True)
    x2.requires_grad_(True)
    
    # objective function
    f = (8 * abs(x1**2 - x2) + (1 - x1)**2)

    # inequality constraint, matrix form
    ci = GeneralStruct()
    ci.c1 = (2**0.5)*x1-1  
    ci.c2 = 2*x2-1 

    # equality constraint 
    ce = None

    return [f,ci,ce]


# ## User Options
# Specify user-defined options for NCVX algorithm

# In[3]:


opts = Options()
# option for switching QP solver. We only have osqp as the only qp solver in current version. Default is osqp
# opts.QPsolver = 'osqp'

# set an intial point
opts.x0 = torch.ones((2,1), device=device, dtype=torch.double)


# ## Main Algorithm

# In[4]:


start = time.time()
soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))
print(soln.final.x)


# ## NCVX Restarting
# **(Optional)** The following example shows how to set NCVX options and how to restart NCVX

# In[5]:


opts = Options()
# set an infeasible initial point
opts.x0 = 5.5*torch.ones((2,1), device=device, dtype=torch.double)

# By default NCVX will print using extended ASCII characters to 'draw' table borders and some color prints. 
# If user wants to create a log txt file of the console output, please set opts.print_ascii = True
opts.print_ascii = True

# By default, NCVX prints an info message about QP solvers, since
# NCVX can be used with any QP solver that has a quadprog-compatible
# interface.  Let's disable this message since we've already seen it 
# hundreds of times and can now recite it from memory.  ;-)
opts.quadprog_info_msg  = False

# Try a very short run. 
opts.maxit = 10 # default is 1000

# NCVX's penalty parameter is on the *objective* function, thus
# higher penalty parameter values favor objective minimization more
# highly than attaining feasibility.  Let's set NCVX to start off
# with a higher initial value of the penalty parameter.  NCVX will
# automatically tune the penalty parameter to promote progress towards 
# feasibility.  NCVX only adjusts the penalty parameter in a
# monotonically decreasing fashion.
opts.mu0 = 100  # default is 1

# start main algorithm
soln = ncvx(combinedFunction = comb_fn, var_dim_map = var_in, torch_device = device, user_opts = opts)


# Let's restart NCVX from the last iterate of the previous run

# In[6]:


opts = Options()
# set the initial point and penalty parameter to their final values from the previous run
opts.x0 = soln.final.x
opts.mu0 = soln.final.mu

# PREPARE TO RESTART NCVX IN FULL-MEMORY MODE
# Set the last BFGS inverse Hessian approximation as the initial
# Hessian for the next run.  Generally this is a good thing to do, and
# often it is necessary to retain this information when restarting (as
# on difficult nonsmooth problems, NCVX may not be able to restart
# without it).  However, your mileage may vary.  In the test, with
# the above settings, omitting H0 causes NCVX to take an additional 
# 16 iterations to converge on this problem. 
opts.H0 = soln.H_final     # try running with this commented out

# When restarting, soln.H_final may fail NCVX's initial check to
# assess whether or not the user-provided H0 is positive definite.  If
# it fails this test, the test may be disabled by setting opts.checkH0 
# to false.
# opts.checkH0 = False       % Not needed for this example 

# If one desires to restart NCVX as if it had never stopped (e.g.
# to continue optimization after it hit its maxit limit), then one must
# also disable scaling the initial BFGS inverse Hessian approximation 
# on the very first iterate. 
opts.scaleH0 = False

# Restart NCVX
opts.maxit = 100 # increase maximum allowed iterations

# Main algorithm
soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)


# In[7]:


soln.final.x


# ## Results Logs
# 
# **(Optional)** opts below shows the importance of using an initial point that is neither near
# nor on a nonsmooth manifold, that is, the functions 
# (objective and constraints) should be smooth at and *about* 
# the initial point.

# In[8]:


opts = Options()
# Set a randomly generated starting point.  In theory, with probability 
# one, a randomly selected point will not be on a nonsmooth manifold.
opts.x0 = torch.randn((2,1), device=device, dtype=torch.double)   # randomly generated is okay
opts.maxit = 100  # we'll use this value of maxit later

# However, (0,0) or (1,1) are on the nonsmooth manifold and if GRANSO
# is started at either of them, it will break down on the first
# iteration.  This example highlights that it is imperative to start
# GRANSO at a point where the functions are smooth.

# Uncomment either of the following two lines to try starting GRANSO
# from (0,0) or (1,1), where the functions are not differentiable. 
    
# opts.x0 = torch.ones((2,1), device=device, dtype=torch.double)     # uncomment this line to try this point
# opts.x0 = torch.zeros((2,1), device=device, dtype=torch.double)    # uncomment this line to try this point

# Uncomment the following two lines to try starting GRANSO from a
# uniformly perturbed version of (1,1).  pert_level needs to be at
# least 1e-3 or so to get consistently reliable optimization quality.

# pert_level = 1e-3
# opts.x0 = (torch.ones((2,1)) + pert_level * (torch.randn((2,1)) - 0.5)).to(device=device, dtype=torch.double)


# The opts below shows how to use opts.halt_log_fn to create a history of iterates
# 
# NOTE: NO NEED TO CHANGE ANYTHING BELOW

# In[9]:


# SETUP THE LOGGING FEATURES
    
# Set up NCVX's logging functions; pass opts.maxit to it so that
# storage can be preallocated for efficiency.

class HaltLog:
    def __init__(self):
        pass

    def haltLog(self, iteration, x, penaltyfn_parts, d,get_BFGS_state_fn, H_regularized,
                ls_evals, alpha, n_gradients, stat_vec, stat_val, fallback_level):

        # DON'T CHANGE THIS
        # increment the index/count 
        self.index += 1                  

        # EXAMPLE:
        # store history of x iterates in a preallocated cell array
        self.x_iterates.append(x)
        self.f.append(penaltyfn_parts.f)
        self.tv.append(penaltyfn_parts.tv)

        # keep this false unless you want to implement a custom termination
        # condition
        halt = False
        return halt
    
    # Once NCVX has run, you may call this function to get retreive all
    # the logging data stored in the shared variables, which is populated 
    # by haltLog being called on every iteration of NCVX.
    def getLog(self):
        # EXAMPLE
        # return x_iterates, trimmed to correct size 
        log = GeneralStruct()
        log.x   = self.x_iterates[0:self.index]
        log.f   = self.f[0:self.index]
        log.tv  = self.tv[0:self.index]
        return log

    def makeHaltLogFunctions(self,maxit):
        # don't change these lambda functions 
        halt_log_fn = lambda iteration, x, penaltyfn_parts, d,get_BFGS_state_fn, H_regularized, ls_evals, alpha, n_gradients, stat_vec, stat_val, fallback_level: self.haltLog(iteration, x, penaltyfn_parts, d,get_BFGS_state_fn, H_regularized, ls_evals, alpha, n_gradients, stat_vec, stat_val, fallback_level)
                
        get_log_fn = lambda : self.getLog()

        # Make your shared variables here to store NCVX history data
        # EXAMPLE - store history of iterates x_0,x_1,...,x_k
        self.index       = 0
        self.x_iterates  = []
        self.f           = []
        self.tv          = []

        # Only modify the body of logIterate(), not its name or arguments.
        # Store whatever data you wish from the current NCVX iteration info,
        # given by the input arguments, into shared variables of
        # makeHaltLogFunctions, so that this data can be retrieved after NCVX
        # has been terminated.
        # 
        # DESCRIPTION OF INPUT ARGUMENTS
        #   iter                current iteration number
        #   x                   current iterate x 
        #   penaltyfn_parts     struct containing the following
        #       OBJECTIVE AND CONSTRAINTS VALUES
        #       .f              objective value at x
        #       .f_grad         objective gradient at x
        #       .ci             inequality constraint at x
        #       .ci_grad        inequality gradient at x
        #       .ce             equality constraint at x
        #       .ce_grad        equality gradient at x
        #       TOTAL VIOLATION VALUES (inf norm, for determining feasibiliy)
        #       .tvi            total violation of inequality constraints at x
        #       .tve            total violation of equality constraints at x
        #       .tv             total violation of all constraints at x
        #       TOTAL VIOLATION VALUES (one norm, for L1 penalty function)
        #       .tvi_l1         total violation of inequality constraints at x
        #       .tvi_l1_grad    its gradient
        #       .tve_l1         total violation of equality constraints at x
        #       .tve_l1_grad    its gradient
        #       .tv_l1          total violation of all constraints at x
        #       .tv_l1_grad     its gradient
        #       PENALTY FUNCTION VALUES 
        #       .p              penalty function value at x
        #       .p_grad         penalty function gradient at x
        #       .mu             current value of the penalty parameter
        #       .feasible_to_tol logical indicating whether x is feasible
        #   d                   search direction
        #   get_BFGS_state_fn   function handle to get the (L)BFGS state data     
        #                       FULL MEMORY: 
        #                       - returns BFGS inverse Hessian approximation 
        #                       LIMITED MEMORY:
        #                       - returns a struct with current L-BFGS state:
        #                           .S          matrix of the BFGS s vectors
        #                           .Y          matrix of the BFGS y vectors
        #                           .rho        row vector of the 1/sty values
        #                           .gamma      H0 scaling factor
        #   H_regularized       regularized version of H 
        #                       [] if no regularization was applied to H
        #   fn_evals            number of function evaluations incurred during
        #                       this iteration
        #   alpha               size of accepted size
        #   n_gradients         number of previous gradients used for computing
        #                       the termination QP
        #   stat_vec            stationarity measure vector                 
        #   stat_val            approximate value of stationarity:
        #                           norm(stat_vec)
        #                       gradients (result of termination QP)
        #   fallback_level      number of strategy needed for a successful step
        #                       to be taken.  See bfgssqpOptionsAdvanced.
        #
        # OUTPUT ARGUMENT
        #   halt                set this to true if you wish optimization to 
        #                       be halted at the current iterate.  This can be 
        #                       used to create a custom termination condition,
        return [halt_log_fn, get_log_fn]

mHLF_obj = HaltLog()
[halt_log_fn, get_log_fn] = mHLF_obj.makeHaltLogFunctions(opts.maxit)

#  Set NCVX's logging function in opts
opts.halt_log_fn = halt_log_fn

# Main algorithm with logging enabled.
soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)

# GET THE HISTORY OF ITERATES
# Even if an error is thrown, the log generated until the error can be
# obtained by calling get_log_fn()
log = get_log_fn()


# In[10]:


print(log.f[0:3])
print(log.x[0:3])


# ## LFBGS Restarting
#  
# **(Optional)**
# 
#  (Note that this example problem only has two variables!)
#  
#  If NCVX was running in limited-memory mode, that is, if 
#  opts.limited_mem_size > 0, then NCVX's restart procedure is 
#  slightly different, as soln.H_final will instead contain the most 
#  current L-BFGS state, not a full inverse Hessian approximation.  
#  
#  Instead, do the following: 
#  1) If you set a specific H0, you will need to set opts.H0 to whatever
#     you used previously.  By default, NCVX uses the identity for H0.
#     
#  2) Warm-start GRANSO with the most recent L-BFGS data by setting:
#     opts.limited_mem_warm_start = soln.H_final;
#     
#  NOTE: how to set opts.scaleH0 so that NCVX will be restarted as if
#  it had never terminated depends on the previously used values of 
#  opts.scaleH0 and opts.limited_mem_fixed_scaling. 

# In[11]:


opts = Options()
# set an infeasible initial point
opts.x0 = 5.5*torch.ones((2,1), device=device, dtype=torch.double)

opts.print_ascii = True
opts.quadprog_info_msg  = False
opts.maxit = 10 # default is 1000
opts.mu0 = 100  # default is 1
opts.print_frequency = 2


# By default, NCVX uses full-memory BFGS updating.  For nonsmooth
# problems, full-memory BFGS is generally recommended.  However, if
# this is not feasible, one may optionally enable limited-memory BFGS
# updating by setting opts.limited_mem_size to a positive integer
# (significantly) less than the number of variables.
opts.limited_mem_size = 1

# start main algorithm
soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)


# In[12]:


# Restart
opts = Options()
# set the initial point and penalty parameter to their final values from the previous run
opts.x0 = soln.final.x
opts.mu0 = soln.final.mu
opts.limited_mem_size = 1
opts.quadprog_info_msg  = False
opts.print_frequency = 2

opts.limited_mem_warm_start = soln.H_final
opts.scaleH0 = False

# In contrast to full-memory BFGS updating, limited-memory BFGS
# permits that H0 can be scaled on every iteration.  By default,
# NCVX will reuse the scaling parameter that is calculated on the
# very first iteration for all subsequent iterations as well.  Set
# this option to false to force NCVX to calculate a new scaling
# parameter on every iteration.  Note that opts.scaleH0 has no effect
# when opts.limited_mem_fixed_scaling is set to true.
opts.limited_mem_fixed_scaling = False

# Restart NCVX
opts.maxit = 100 # increase maximum allowed iterations

# Main algorithm
soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)

