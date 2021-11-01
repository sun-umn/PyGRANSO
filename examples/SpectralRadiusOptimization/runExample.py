import time
import torch
import os,sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso import pygranso
from pygransoStruct import Options, Data
import scipy.io

# Please read the documentation on https://pygranso.readthedocs.io/en/latest/

# device = torch.device('cuda' )
device = torch.device('cpu' )
print('Using device:', device)

# read input data from matlab file
currentdir = os.path.dirname(os.path.realpath(__file__))
file = currentdir + "/spec_radius_opt_data.mat"
mat = scipy.io.loadmat(file)
mat_struct = mat['sys']
mat_struct = mat_struct[0,0]
A = torch.from_numpy(mat_struct['A']).to(device=device, dtype=torch.double)
B = torch.from_numpy(mat_struct['B']).to(device=device, dtype=torch.double)
C = torch.from_numpy(mat_struct['C']).to(device=device, dtype=torch.double)
p = B.shape[1]
m = C.shape[0]

# variable and corresponding dimensions
var_in = {"X": (p,m) }

# data_in
data_in = Data()
data_in.A = A
data_in.B = B
data_in.C = C
data_in.stability_margin = 1

# user defined options
opts = Options()
opts.QPsolver = 'osqp' 
opts.maxit = 200
opts.x0 = torch.zeros(p*m,1).to(device=device, dtype=torch.double)
opts.print_level = 1
opts.print_frequency = 1
opts.limited_mem_size = 40

#  main algorithm

start = time.time()
soln = pygranso(var_dim_map = var_in, torch_device=device, user_data = data_in, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))
pass
