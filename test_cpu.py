import time
import torch

from ncvx import ncvx
from ncvxStruct import Options, GeneralStruct 
import scipy.io
from torch import linalg as LA
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
import sys
## Adding NCVX directories. Should be modified by user
sys.path.append(currentdir)
import numpy as np

device = torch.device('cpu')
from scipy.stats import norm
import numpy.linalg as la




def rosenbrock():
    # variables and corresponding dimensions.
    var_in = {"x1": [1,1], "x2": [1,1]}

    def comb_fn(X_struct):
        x1 = X_struct.x1
        x2 = X_struct.x2
        # enable autodifferentiation
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        
        # objective function
        f = (8 * abs(x1**2 - x2) + (1 - x1)**2)

        # inequality constraint, matrix form
        ci = GeneralStruct()
        ci.c1 = (2**0.5)*x1-1  
        ci.c2 = 2*x2-1 

        # equality constraint 
        ce = None

        return [f,ci,ce]

    opts = Options()
    # option for switching QP solver. We only have osqp as the only qp solver in current version. Default is osqp
    # opts.QPsolver = 'osqp'

    # set an intial point
    opts.x0 = torch.ones((2,1), device=device, dtype=torch.double)
    opts.print_level = 0


    soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
    print("test 1/7 passed (Rosenbrock)")

def spectral_radius():

    
    file = "{}/examples/data/spec_radius_opt_data.mat".format(currentdir)
    mat = scipy.io.loadmat(file)
    mat_struct = mat['sys']
    mat_struct = mat_struct[0,0]
    A = torch.from_numpy(mat_struct['A']).to(device=device, dtype=torch.double)
    B = torch.from_numpy(mat_struct['B']).to(device=device, dtype=torch.double)
    C = torch.from_numpy(mat_struct['C']).to(device=device, dtype=torch.double)
    p = B.shape[1]
    m = C.shape[0]
    stability_margin = 1

    # variables and corresponding dimensions.
    var_in = {"X": [p,m] }

    def comb_fn(X_struct):
        # user defined variable, matirx form. torch tensor
        X = X_struct.X
        X.requires_grad_(True)

        # objective function
        M           = A + B@X@C
        [D,_]       = LA.eig(M)
        f = torch.max(D.imag)

        # inequality constraint, matrix form
        ci = GeneralStruct()
        ci.c1 = torch.max(D.real) + stability_margin

        # equality constraint 
        ce = None
        
        return [f,ci,ce]

    opts = Options()
    opts.maxit = 10
    opts.x0 = torch.zeros(p*m,1).to(device=device, dtype=torch.double)
    # print for every 10 iterations. default: 1
    opts.print_frequency = 10
    opts.print_level = 0

    soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
    print("test 2/7 passed (Spectral Radius Optimization)")

def dictionary_learning():


    n = 30

    np.random.seed(1)
    m = 10*n**2   # sample complexity
    theta = 0.3   # sparsity level
    Y = norm.ppf(np.random.rand(n,m)) * (norm.ppf(np.random.rand(n,m)) <= theta)  # Bernoulli-Gaussian model
    Y = torch.from_numpy(Y).to(device=device, dtype=torch.double)

    # variables and corresponding dimensions.
    var_in = {"q": [n,1]}


    def comb_fn(X_struct):
        q = X_struct.q
        q.requires_grad_(True)

        # objective function
        qtY = q.T @ Y
        f = 1/m * torch.norm(qtY, p = 1)

        # inequality constraint, matrix form
        ci = None

        # equality constraint
        ce = GeneralStruct()
        ce.c1 = q.T @ q - 1

        return [f,ci,ce]

    opts = Options()
    opts.QPsolver = 'osqp'
    opts.maxit = 20
    np.random.seed(1)
    x0 = norm.ppf(np.random.rand(n,1))
    x0 /= la.norm(x0,2)
    opts.x0 = torch.from_numpy(x0).to(device=device, dtype=torch.double)
    opts.print_level = 0


    opts.print_frequency = 10

    soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
    print("test 3/7 passed (Dictionary Learning)")

def robust_PCA():
    d1 = 3
    d2 = 4
    torch.manual_seed(1)
    eta = .05
    Y = torch.randn(d1,d2).to(device=device, dtype=torch.double)

    # variables and corresponding dimensions.
    var_in = {"M": [d1,d2],"S": [d1,d2]}


    opts = Options()
    opts.print_frequency = 10
    opts.x0 = .2 * torch.ones((2*d1*d2,1)).to(device=device, dtype=torch.double)
    opts.opt_tol = 1e-6
    opts.maxit = 50
    opts.print_level = 0

    def comb_fn(X_struct):
        M = X_struct.M
        S = X_struct.S
        M.requires_grad_(True)
        S.requires_grad_(True)
        
        # objective function
        f = torch.norm(M, p = 'nuc') + eta * torch.norm(S, p = 1)

        # inequality constraint, matrix form
        ci = None
        
        # equality constraint 
        ce = GeneralStruct()
        ce.c1 = M + S - Y

        return [f,ci,ce]

    soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
    print("test 4/7 passed (Robust PCA)")

def lasso():
    n = 80
    eta = 0.5 # parameter for penalty term
    torch.manual_seed(1)
    b = torch.rand(n,1)
    pos_one = torch.ones(n-1)
    neg_one = -torch.ones(n-1)
    F = torch.zeros(n-1,n)
    F[:,0:n-1] += torch.diag(neg_one,0) 
    F[:,1:n] += torch.diag(pos_one,0)
    F = F.to(device=device, dtype=torch.double)  # double precision requireed in torch operations 
    b = b.to(device=device, dtype=torch.double)

    # variables and corresponding dimensions.
    var_in = {"x": [n,1]}


    def comb_fn(X_struct):
        x = X_struct.x
        x.requires_grad_(True)
        
        # objective function
        f = (x-b).t() @ (x-b)  + eta * torch.norm( F@x, p = 1)
        
        # inequality constraint, matrix form
        ci = None
        # equality constraint 
        ce = None

        return [f,ci,ce]

    opts = Options()
    opts.QPsolver = 'osqp' 
    opts.x0 = torch.ones((n,1)).to(device=device, dtype=torch.double)
    opts.print_level = 1
    opts.print_frequency = 10
    opts.print_level = 0
    opts.maxit = 30
    soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)

    print("test 5/7 passed (LASSO)")

def feasibility():
    # variables and corresponding dimensions.
    var_in = {"x": [1,1],"y": [1,1]}


    def comb_fn(X_struct):
        x = X_struct.x
        y = X_struct.y
        x.requires_grad_(True)
        y.requires_grad_(True)
        # constant objective function
        f = 0*x+0*y

        # inequality constraint 
        ci = GeneralStruct()
        ci.c1 = (y+x**2)**2+0.1*y**2-1
        ci.c2 = y - torch.exp(-x) - 3
        ci.c3 = y-x+4
        
        # equality constraint 
        ce = None

        return [f,ci,ce]

    opts = Options()
    opts.QPsolver = 'osqp' 
    opts.print_frequency = 1
    opts.x0 = 0 * torch.ones((2,1)).to(device=device, dtype=torch.double)
    opts.print_level = 0

    soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
    print("test 6/7 passed (Feasibility Problem)")


def sphere_manifold():

    torch.manual_seed(0)
    n = 500
    A = torch.randn((n,n)).to(device=device, dtype=torch.double)
    A = .5*(A+A.T)

    # variables and corresponding dimensions.
    var_in = {"x": [n,1]}


    def comb_fn(X_struct):
        x = X_struct.x
        x.requires_grad_(True)

        # objective function
        f = x.T@A@x

        # inequality constraint, matrix form
        ci = None

        # equality constraint
        ce = GeneralStruct()
        ce.c1 = x.T@x-1

        return [f,ci,ce]

    opts = Options()
    opts.print_frequency = 10
    opts.x0 = torch.randn((n,1)).to(device=device, dtype=torch.double)
    opts.mu0 = 0.1 # increase penalty contribution
    opts.opt_tol = 1e-6
    opts.maxit = 20
    opts.print_level = 0

    soln = ncvx(combinedFunction = comb_fn,var_dim_map = var_in, torch_device = device, user_opts = opts)
    print("test 7/7 passed (Sphere Manifold)")



if __name__ == "__main__" :
    count = 1
    try:
        print("Testing may take several minutes. Thank you for your patience.")
        rosenbrock()
        count += 1
        spectral_radius()
        count += 1
        dictionary_learning()
        count += 1
        robust_PCA()
        count += 1
        lasso()
        count += 1
        feasibility()
        count += 1
        sphere_manifold()
        print("Successfully passed all tests!")
    except Exception:
        print("Test {} fail, please carefully read the instructions on https://ncvx.org/ for installation".format(count))
    