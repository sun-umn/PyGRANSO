import time
from typing import Tuple
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from torch.linalg import norm
from scipy.stats import ortho_group
import numpy as np
from matplotlib import pyplot as plt 
import os
from datetime import datetime


###############################################
n = 10 # V: n*d
d = 5 # copnst: d*d
# maxfolding = 'unfolding'
maxfolding = 'l2'
total = 100 # total number of starting points
feasible_init = True
opt_tol = 1e-6
maxit = 1000
maxclocktime = 20
# QPsolver = "gurobi"
QPsolver = "osqp"
###############################################

device = torch.device('cuda')
torch.manual_seed(2023)

A = torch.randn(n,n)
A = (A + A.T)/2
# All the user-provided data (vector/matrix/tensor) must be in torch tensor format.
# As PyTorch tensor is single precision by default, one must explicitly set `dtype=torch.double`.
# Also, please make sure the device of provided torch tensor is the same as opts.torch_device.
A = A.to(device=device, dtype=torch.double)

L, U = torch.linalg.eig(A)
L = L.to(dtype=torch.double)
U = U.to(dtype=torch.double)
index = torch.argsort(L,descending=True)
U = U[:,index[0:d]]


F_lst = [ -11.3850734  , -11.42030888 , -11.44291946 , -11.46321246
           , -11.53502873 , -11.59205934 , -11.88270666 , -11.94352778 , -11.94699448
           , -11.97244151 , -11.9947808  , -12.00162411 , -12.02222239 , -12.09298062
           , -12.10857034 , -12.20008112 , -12.21933422 , -12.22290175 , -12.2252395
           , -12.23978361 , -12.24563256 , -12.26081238 , -12.2660355  , -12.27239271
           , -12.27313955 , -12.27813205 , -12.28131294 , -12.28215878 , -12.28739973
           , -12.29403087 , -12.30043521 , -12.30044686 , -12.30639651 , -12.31126727
           , -12.3123256  , -12.31256104 , -12.31963122 , -12.33320901 , -12.33621202
           , -12.34333682 , -12.34850717 , -12.35086177 , -12.35483678 , -12.35484298
           , -12.35626432 , -12.35682614 , -12.35706318 , -12.36414911 , -12.36807015
           , -12.36894126 , -12.37021325 , -12.37182555 , -12.37193099 , -12.37231546
           , -12.37248612 , -12.37250329 , -12.37259102 , -12.37285456 , -12.37298449
           , -12.37307021 , -12.37352268 , -12.3735244  , -12.37419209 , -12.37424502
           , -12.37432961 , -12.37433355 , -12.37433421 , -12.37438793 , -12.37442196
           , -12.3744415  , -12.37447094 , -12.37447697 , -12.3744854  , -12.37448561
           , -12.37448875 ]
MF_lst = [ -11.38363891, -11.42019729, -11.4428764 , -11.46255092
, -11.53449079, -11.5911229 , -11.88067066, -11.94325098, -11.9453604
, -11.97213507, -11.99287846, -12.00148548, -12.02014816, -12.09282419
, -12.10782752, -12.19926992, -12.21874326, -12.22161789, -12.22415608
, -12.23954542, -12.24477312, -12.26047562, -12.26352485, -12.27056159
, -12.2730575 , -12.27590533, -12.28107478, -12.28082719, -12.28739973
, -12.29393571, -12.30030594, -12.29865244, -12.30599812, -12.31021689
, -12.31218429, -12.31199786, -12.31851483, -12.33168818, -12.33474868
, -12.34241902, -12.3484422 , -12.35010117, -12.35466696, -12.35460691
, -12.35555415, -12.35647751, -12.35648823, -12.36384419, -12.36756673
, -12.36888801, -12.3702112 , -12.37158786, -12.37156232, -12.37229286
, -12.37247904, -12.37241713, -12.37254961, -12.37283441, -12.37281425
, -12.37299156, -12.37335232, -12.37335955, -12.37418389, -12.37420662
, -12.37431675, -12.37433057, -12.37432086, -12.37437423, -12.37440788
, -12.37441178, -12.37446608, -12.37447693, -12.37448316, -12.37448435
, -12.37448816  ]

F_arr = np.array(F_lst)
MF_arr = np.array(MF_lst)

arr_len = F_arr.shape[0]
plt.plot(np.arange(arr_len),F_arr,'ro-',label='sorted_pygranso_sol_F')
plt.plot(np.arange(arr_len),MF_arr,'go-',label='sorted_pygranso_sol_MF')

ana_sol = torch.trace(U.T@A@U).item()
plt.plot(np.arange(arr_len),np.array(arr_len*[-ana_sol]),'b-',label='analytical_sol')
plt.legend()
# plt.show()

now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y_%H:%M:%S")
if feasible_init:
    feasible = "feasible_init"
else:
    feasible = "gaussnormal_init"


png_title =  "png/sorted_F_" + date_time + "_n{}_d{}_{}_{}".format(n,d,feasible,maxfolding)

my_path = os.path.dirname(os.path.abspath(__file__))

plt.savefig(os.path.join(my_path, png_title))