from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import datasets
from sklearn.svm import l1_min_c

import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
import torch

device = torch.device('cuda')

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 2]
y = y[y != 2]

X /= X.max()  # Normalize X to speed-up convergence

# Demo path functions

cs = l1_min_c(X, y, loss="log") * np.logspace(0, 7, 16)
# cs = l1_min_c(X, y, loss="log") * np.logspace(0, 5, 4)


X = torch.from_numpy(X).to(device=device, dtype=torch.double)
y = torch.from_numpy(y).to(device=device, dtype=torch.double)
n = y.shape[0]
y = y.unsqueeze(1)
# #############################################################################





# variables and corresponding dimensions.
var_in = {"w": [4,1]}


def user_fn(X_struct,X,y,C):
    w = X_struct.w

    # objective function
    # for i in range(n):
    #     if i == 0:
    #         f = torch.log(torch.exp(-y[i]* X[i,:]@w) + 1)
    #     else:
    #         f += torch.log(torch.exp(-y[i]* X[i,:]@w) + 1)

    # torch.reshape(y,(n,1))
    
    f = torch.sum(torch.log(torch.exp(-y* (X@w)) + 1))
    f+= torch.norm(w,p=1)/C

    # inequality constraint, matrix form
    ci = None

    # equality constraint
    ce = None

    return [f,ci,ce]




print("Computing regularization path ...")
start = time()
# clf = linear_model.LogisticRegression(
#     penalty="l1",
#     solver="liblinear",
#     tol=1e-6,
#     max_iter=int(1e6),
#     warm_start=True,
#     intercept_scaling=10000.0,
# )

opts = pygransoStruct()
opts.torch_device = device
opts.maxit = 50
opts.opt_tol = 1e-6
np.random.seed(1)
opts.x0 = torch.zeros(4,1).to(device=device, dtype=torch.double)
# torch.manual_seed(0)
# opts.x0 = torch.randn(4, 1).to(device=device, dtype=torch.double)
# opts.print_frequency = 10

coefs_ = []
i = 0
for c in cs:
    print("c = {}, i = {}".format(c,i))
    i += 1
    comb_fn = lambda X_struct : user_fn(X_struct,X,y,c)
    torch.autograd.set_detect_anomaly(True)
    soln = pygranso(var_spec = var_in,combined_fn = comb_fn,user_opts = opts)
    arr = soln.final.x.T.tolist()
    arr = np.array(arr).ravel()
    
    coefs_.append(arr)
print("This took %0.3fs" % (time() - start))

coefs_ = np.array(coefs_)
plt.plot(np.log10(cs), coefs_, marker="o")
ymin, ymax = plt.ylim()
plt.xlabel("log(C)")
plt.ylabel("Coefficients")
plt.title("Logistic Regression Path")
plt.axis("tight")
plt.show()