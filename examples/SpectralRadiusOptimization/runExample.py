import time
import numpy as np
import torch
import numpy.linalg as la
from scipy.stats import norm
import os,sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append(r'C:\Users\Buyun\Documents\GitHub\PyGRANSO')
from pygranso import pygranso
from pygransoStruct import Options, Parameters
import scipy.io

# Please read the documentation on https://pygranso.readthedocs.io/en/latest/

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# parameters
parameters = Parameters()
parameters.A = A
parameters.B = B
parameters.C = C
parameters.stability_margin = 1

# user defined options
opts = Options()
opts.QPsolver = 'osqp' 
opts.maxit = 20
# opts.x0 = np.zeros((p*m,1))
opts.x0 = torch.zeros(p*m,1).to(device=device, dtype=torch.double)
opts.print_level = 1
opts.print_frequency = 1

#  main algorithm

start = time.time()
soln = pygranso(var_in,parameters,opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))
pass
