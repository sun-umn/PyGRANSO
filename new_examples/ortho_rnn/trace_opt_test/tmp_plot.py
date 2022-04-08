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
write_to_log = True

n = 30 # V: n*d
d = 15 # copnst: d*d
maxfolding = 'unfolding'
# maxfolding = 'l2'
# maxfolding = 'l1'
# maxfolding = 'linf'

total = 100 # total number of starting points
feasible_init = False
opt_tol = 1e-6
maxit = 1000
maxclocktime = 100
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


F_lst = [-46.69061003, -46.98557115, -47.0849325 , -47.14932051, -47.46973308
, -47.51212489, -47.61474762, -47.70669027, -47.73369192, -47.73771633
, -47.81942951, -47.83239835, -47.89981505, -47.91127987, -47.98154597
, -48.0227934 , -48.03357154, -48.14793096, -48.16555115, -48.17112539
, -48.17255349, -48.19576427, -48.21220564, -48.21527149, -48.23586758
, -48.26301006, -48.28510603, -48.3049819 , -48.32656826, -48.36244047
, -48.38129803, -48.38380133, -48.40929492, -48.4378786 , -48.44990177
, -48.47018526, -48.47168783, -48.50393666, -48.50799779, -48.51090646
, -48.52927058, -48.53628414, -48.53956549, -48.54310481, -48.54454036
, -48.5636696 , -48.58331777, -48.6063569 , -48.64388024, -48.65388887
, -48.70112729, -48.71072701, -48.71455013, -48.71651829, -48.75164882
, -48.76516977, -48.77783259, -48.78805457, -48.78976611, -48.80984835
, -48.81472507, -48.82964654, -48.84871514, -48.87804019, -48.88602952
, -48.88793586, -48.8927376 , -48.90466988, -48.93073143, -48.93934766
, -48.97684316, -48.9870224 , -48.99270116, -49.00793145, -49.00977278
, -49.02800432, -49.0341883 , -49.05116897, -49.06837395, -49.08151598
, -49.08198792, -49.09754083, -49.10066577, -49.12908366, -49.1388517
, -49.17990691, -49.18191685, -49.18474959, -49.18476582, -49.19354651
, -49.19621067, -49.22618861]
MF_lst = [-46.69061003, -46.98557092, -47.0849325 , -47.1492916 , -47.46921654
, -47.51197455, -47.61474762, -47.70565019, -47.73356667, -47.73766574
, -47.81942951, -47.83239835, -47.89976031, -47.91120899, -47.98147721
, -48.02265272, -48.03343075, -48.14793096, -48.16528487, -48.17112901
, -48.17245045, -48.1957645 , -48.21214247, -48.21514112, -48.23586735
, -48.26301006, -48.28488904, -48.30468423, -48.32608955, -48.36244047
, -48.38124124, -48.38380133, -48.40929492, -48.43787913, -48.44986474
, -48.47018526, -48.47168629, -48.50390021, -48.50799779, -48.51090646
, -48.52925364, -48.53628406, -48.53956549, -48.54305755, -48.54454036
, -48.56362388, -48.58331777, -48.60612206, -48.64371448, -48.65378519
, -48.70112729, -48.71072739, -48.71452674, -48.71629282, -48.75164882
, -48.76499419, -48.77780374, -48.78805457, -48.78976561, -48.80984835
, -48.8147401 , -48.8296464 , -48.84843871, -48.8780398 , -48.8860265
, -48.88763928, -48.8927376 , -48.90466988, -48.93070719, -48.93934766
, -48.97666184, -48.98598733, -48.99253003, -49.00773342, -49.00977278
, -49.02799691, -49.03418896, -49.05109418, -49.06812597, -49.08144447
, -49.08192092, -49.09743251, -49.10065   , -49.12908366, -49.13876533
, -49.1798642 , -49.18191242, -49.18473029, -49.18369885, -49.19354651
, -49.19621069, -49.22618372]

F_arr = np.array(F_lst)
MF_arr = np.array(MF_lst)

arr_len = F_arr.shape[0]
plt.plot(np.arange(arr_len),F_arr,'ro-',label='sorted_pygranso_sol_F')
plt.plot(np.arange(arr_len),MF_arr,'go-',label='sorted_pygranso_sol_MF')

ana_sol = torch.trace(U.T@A@U).item()
plt.plot(np.arange(arr_len),np.array(arr_len*[-ana_sol]),'b-',label='analytical_sol')
plt.legend()
plt.ylim([-50, -16])
# plt.show()

now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y_%H:%M:%S")
if feasible_init:
    feasible = "feasible_init"
else:
    feasible = "gaussnormal_init"


png_title =  "png/Modified_png_" + date_time + "_n{}_d{}_{}_{}".format(n,d,feasible,maxfolding)

my_path = os.path.dirname(os.path.abspath(__file__))

plt.title("n{}_d{}_{}_{}".format(n,d,feasible,maxfolding))


plt.savefig(os.path.join(my_path, png_title))