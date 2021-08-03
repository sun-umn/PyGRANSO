import time
import numpy as np
from numpy.random import default_rng
import torch

n=500
m=10*n**2

A = np.ones((n,m))
q1  = np.ones((n,1))

B = torch.ones(n,m).double()
q2 = torch.ones(n,1).double()

start = time.time()
A @ A.T@q1
end = time.time()
print("numpy: ATq: Total Wall Time is {}s".format(end - start))

start = time.time()
B @ B.t() @ q2
end = time.time()
print("torch: BTq: Total Wall Time is {}s".format(end - start))

# rng = default_rng()
# # A = np.random.randn(2**13,2**13)
# # B = np.random.randn(2**12,2**11)
# # F = torch.randn(2**13,2**13)

# start = time.time()
# C = np.random.randn(2**13,2**13)
# end = time.time()
# print("generate 8192 by 8192 matrix: Total Wall Time is {}s".format(end - start))

# start = time.time()
# D = rng.standard_normal(size=(2**13,2**13))
# end = time.time()
# print("New version: generate 8192 by 8192 matrix: Total Wall Time is {}s".format(end - start))

# start = time.time()
# E = torch.randn(2**13,2**13)
# end = time.time()
# print("Torch: generate 8192 by 8192 matrix: Total Wall Time is {}s".format(end - start))



# start = time.time()
# np.linalg.svd(B)
# end = time.time()
# print("SVD of a 4096 by 2048 matrix: Total Wall Time is {}s".format(end - start))





pass