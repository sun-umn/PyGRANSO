import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from pygranso.private.getObjGrad import getObjGradDL
from torch.linalg import norm



sys.path.append('/home/buyun/Documents/GitHub/deepxde')

import deepxde as dde
import numpy as np
from scipy.io import loadmat
from deepxde import losses as losses_module
from deepxde import gradients as grad


torch.manual_seed(2022)
np.random.seed(2022)
dde.config.set_random_seed(2022)

device = torch.device('cuda')
# device = torch.device('cpu')

double_precision = torch.double

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
d = 0.001


def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - d * dy_xx - 5 * (y - y**3)


def gen_testdata():
    data = loadmat("/home/buyun/Documents/GitHub/deepxde/examples/dataset/Allen_Cahn.mat")

    t = data["t"]
    x = data["x"]
    u = data["u"]

    dt = dx = 0.01
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]
    return X, y


class FNN(torch.nn.Module):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes):
        super().__init__()
        # self.activation = activations.get(activation)
        # initializer = initializers.get(kernel_initializer)
        # initializer_zero = initializers.get("zeros")

        self.activation = torch.tanh
        # self.activation = torch.nn.functional.relu


        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i]
                )
            )
            # initializer(self.linears[-1].weight)
            # initializer_zero(self.linears[-1].bias)

    def forward(self, inputs):
        x = inputs
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        return x



# define torch network 
model = FNN([2] + [20] * 3 + [1])
model = model.to(device=device, dtype=double_precision)

# get data
data = dde.data.TimePDE(geomtime, pde, [], num_domain=8000, num_boundary=400, num_initial=800)

X_train = torch.from_numpy(data.train_x)
# y_train = torch.from_numpy(data.train_y)
X_train = X_train.to(device=device, dtype=double_precision)
# y_train = y_train.to(device=device, dtype=double_precision)


X_test, y_test = gen_testdata()
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)
X_test = X_test.to(device=device, dtype=double_precision)
y_test = y_test.to(device=device, dtype=double_precision)


def user_fn(model,X_train):
    
    X_train.requires_grad_()
    y_predict = model(X_train) # graph of X_train

    f_vec = pde(X_train,y_predict)
    f = norm(f_vec,2)

    # inequality constraint
    ci = None

    # equality constraint
    # special orthogonal group

    ce = pygransoStruct()
    mask_t0 = (X_train[:,1:2]==0).squeeze()
    X_t0 = X_train[mask_t0] # when t = 0
    y_predict_t0 = y_predict[mask_t0]
    # ce.c1 = y_predict_t0 - X_t0[:,0:1]**2 * torch.cos(np.pi * X_t0[:, 0:1])
    constr_1 = y_predict_t0 - X_t0[:,0:1]**2 * torch.cos(np.pi * X_t0[:, 0:1])

    mask_pn1 = torch.logical_or(X_train[:,0:1]==1,X_train[:,0:1]==-1).squeeze()

    X_pn1 = X_train[mask_pn1] # when x = +/-1
    # ce.c2 = model(X_pn1) +1
    constr_2 = model(X_pn1) +1

    # ce.c1 = norm(torch.vstack((constr_1,constr_2)),2)

    ce.c1 = norm(constr_1,2)
    ce.c2 = norm(constr_2,2)

    grad.clear() # important: please clear jacobian and hessian in deepxde!
    return [f,ci,ce]

comb_fn = lambda model : user_fn(model,X_train)

opts = pygransoStruct()
opts.torch_device = device
nvar = getNvarTorch(model.parameters())
opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
# opts.opt_tol = 3e-4
# opts.viol_eq_tol = 3e-4
opts.maxit = 1000
# # opts.fvalquit = 1e-6
# opts.print_level = 1
opts.print_frequency = 10
# # opts.print_ascii = True
# opts.limited_mem_size = 10
# opts.double_precision = True

# opts.mu0 = 200


# _, predicted = torch.max(logits.data, 1)
# correct = (predicted == labels).sum().item()
# print("Initial acc = {:.2f}%".format((100 * correct/len(inputs))))


criterion = nn.MSELoss()

# l2_rel_err = norm(model(X_train)-y_train)/norm(y_train)
# print("intial y_train l2_rel_err = {}".format(l2_rel_err))

# test_loss = criterion(model(X_test),y_test)
# print("initial test mse loss = {}".format(test_loss))


# from torch.linalg import norm
# l2_rel_err = norm(model(X_test)-y_test)/norm(y_test)
# print("initial test l2_rel_err = {}".format(l2_rel_err))

l2_rel_err = norm(model(X_test)-y_test)/norm(y_test)
print("initial test l2_rel_err = {}".format(l2_rel_err))


start = time.time()
soln = pygranso(var_spec= model, combined_fn = comb_fn, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))

torch.nn.utils.vector_to_parameters(soln.final.x, model.parameters())
# logits = model(inputs)
# _, predicted = torch.max(logits.data, 1)
# correct = (predicted == labels).sum().item()
# print("Final acc = {:.2f}%".format((100 * correct/len(inputs))))

# test_loss = criterion(model(X_test),y_test)
# print("final test mse loss = {}".format(test_loss))

# # def l2_relative_error(y_true, y_pred):
# #     return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)

# from torch.linalg import norm
# l2_rel_err = norm(model(X_test)-y_test)/norm(y_test)
# print("test l2_rel_err = {}".format(l2_rel_err))

# l2_rel_err = norm(model(X_train)-y_train)/norm(y_train)
# print("y_train l2_rel_err = {}".format(l2_rel_err))






# print("Mean residual:", np.mean(np.absolute(f)))
# print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
l2_rel_err = norm(model(X_test)-y_test)/norm(y_test)
print("final test l2_rel_err = {}".format(l2_rel_err))