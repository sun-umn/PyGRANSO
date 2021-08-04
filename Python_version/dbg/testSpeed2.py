import time
import numpy as np
import numpy.linalg as la
from numpy.random import default_rng
import torch

n=200
m=10*n**2

theta = .3

Y2 = torch.randn(n,m) * (torch.rand(n,m) <= theta) # Bernoulli-Gaussian model
Y1  = Y2.numpy()

A = np.ones((n,m))
q1  = np.ones((n,1))

B = torch.ones(n,m).double()
q2 = torch.ones(n,1).double()

start = time.time()
qtY = Y1.T @ q1
f1 = 1/m * la.norm(qtY,  ord = 1)
end = time.time()
print("numpy: ATq: Total Wall Time is {}s".format(end - start))

start = time.time()
qtY = q2.t() @ Y2
f2 = 1/m * torch.norm(qtY, p = 1)
end = time.time()
print("torch: BTq: Total Wall Time is {}s".format(end - start))


pass