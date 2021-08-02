import time
import numpy as np
from numpy.random import default_rng
import torch

rng = default_rng()
A = np.random.randn(2**13,2**13)
B = np.random.randn(2**12,2**11)


start = time.time()
C = np.random.randn(2**13,2**13)
end = time.time()
print("generate 8192 by 8192 matrix: Total Wall Time is {}s".format(end - start))

start = time.time()
D = rng.standard_normal(size=(2**13,2**13))
end = time.time()
print("New version: generate 8192 by 8192 matrix: Total Wall Time is {}s".format(end - start))

start = time.time()
E = torch.randn(2**13,2**13)
end = time.time()
print("Torch: generate 8192 by 8192 matrix: Total Wall Time is {}s".format(end - start))

start = time.time()
A.T@A
end = time.time()
print("Dotted two 8192 by 8192 matrix: Total Wall Time is {}s".format(end - start))

start = time.time()
np.linalg.svd(B)
end = time.time()
print("SVD of a 4096 by 2048 matrix: Total Wall Time is {}s".format(end - start))





pass