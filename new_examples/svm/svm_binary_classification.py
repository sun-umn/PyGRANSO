import time
from pyrsistent import b
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch
import torch.nn as nn
from torchvision.transforms import ToTensor

from pygranso.private.getObjGrad import getObjGradDL

from torch.linalg import norm

from sklearn import datasets



device = torch.device('cuda')



iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 2]
y = y[y != 2]
y[y==0] = -1

X /= X.max()  # Normalize X to speed-up convergence
# X = X[0:10,:]
# y = y[0:10]

##############################################################

# import numpy as np
# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, -1, -1])
# # X = np.array([[-1, -1],  [1, 1]])
# # y = np.array([-1,1])

# from sklearn.svm import SVC
# clf = SVC(kernel='linear')
# clf.fit(X, y)
# prediction = clf.predict([[0,6]])


##############################################################

# bc = datasets.load_breast_cancer()
# X = bc.data
# y = bc.target
# y[y==0] = -1
# X /= X.max()  # Normalize X to speed-up convergence


X = torch.from_numpy(X).to(device=device, dtype=torch.double)
y = torch.from_numpy(y).to(device=device, dtype=torch.double)
[n,d] = X.shape
y = y.unsqueeze(1)

zeta = 0.00

# variables and corresponding dimensions.
# var_in = {"w": [d,1], "zeta": [n,1], "b": [n,1], "C": [1,1]}
var_in = {"w": [d,1], "b": [1,1]}


def user_fn(X_struct,X,y, zeta):
    w = X_struct.w
    b = X_struct.b    
    f = 0.5*w.T@w 

    # inequality constraint 
    ci = pygransoStruct()

    constr = 1 - zeta - y*(X@w+b)

    # ci.c1 = torch.sum(torch.clamp(constr, min=0)) # l1
    ci.c1 = torch.sum(torch.clamp(constr, min=0)**2)**0.5 # l2
    # ci.c1 = torch.max(constr) # l_inf



    # ci.c1 = constr

    # equality constraint
    ce = None

    # # inequality constraint 2
    # ci = pygransoStruct()
    # # constr = torch.zeros((2*n,1)).to(device=device, dtype=torch.double)
    # # constr[0:n] = 1 - zeta - y*(X@w+b)
    # # constr[n:2*n] = (1 - zeta)**2 - (y*(X@w+b))**2 # dummy nonlinear constr

    return [f,ci,ce]


comb_fn = lambda X_struct : user_fn(X_struct,X,y,zeta)

opts = pygransoStruct()
opts.torch_device = device

torch.manual_seed(2022)
opts.x0 = torch.randn(d+1,1).to(device=device, dtype=torch.double)
opts.opt_tol = 1e-5
opts.viol_eq_tol = 1e-5
opts.maxit = 60000
# opts.fvalquit = 1e-6
opts.print_level = 1
opts.print_frequency = 20
# opts.print_ascii = True
# opts.limited_mem_size = 100
opts.double_precision = True

# opts.maxclocktime = 440

# opts.steering_c_viol = 0.9

# opts.mu0 = 1e-4

# opts.steering_c_mu = 0.5

# opts.globalAD = False # disable global auto-differentiation




# logits = model(inputs)
# _, predicted = torch.max(logits.data, 1)
# correct = (predicted == labels).sum().item()
# print("Initial acc = {:.2f}%".format((100 * correct/len(inputs))))


start = time.time()
soln = pygranso(var_spec = var_in,combined_fn = comb_fn,user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))


w = soln.final.x[0:d]
b = soln.final.x[d:d+1]
# zeta = soln.final.x[d:d+n]
res = X@w+b
predicted = torch.zeros(n,1).to(device=device, dtype=torch.double)
predicted[res>=0] = 1
predicted[res<0] = -1
correct = (predicted == y).sum().item()
print("Final acc = {:.2f}%".format((100 * correct/n)))

# print('zeta = {}'.format(zeta))
print('w = {}'.format(w))
print('b = {}'.format(b))