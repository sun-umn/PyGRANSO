import time
import torch

n=500
m=10*n**2
theta = .3

Y = torch.ones(n,m)
q = torch.ones(n,1)

start = time.time()
output = Y @ Y.t() @ q
end = time.time()
print("Torch Y @ Y.t() @ q: Total Wall Time is {}s".format(end - start))





pass