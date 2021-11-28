import os
currentdir = os.path.dirname(os.path.realpath(__file__))

def spectral_radius():
    import time
    import torch
    import os,sys
    ## Adding NCVX directories. Should be modified by user
    sys.path.append(currentdir)
    from ncvx import ncvx
    from ncvxStruct import Options, Data, GeneralStruct 
    import scipy.io
    from torch import linalg as LA

    device = torch.device('cuda')

    
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
    print("test 1/3 passed")

def dictionary_learning():
    import time
    import numpy as np
    import torch
    import numpy.linalg as la
    from scipy.stats import norm
    import sys
    ## Adding NCVX directories. Should be modified by user
    sys.path.append(currentdir)
    from ncvx import ncvx
    from ncvxStruct import Options, GeneralStruct

    device = torch.device('cuda')
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
    print("test 2/3 passed")

def sphere_manifold():
    import time
    import torch
    import sys
    ## Adding NCVX directories. Should be modified by user
    sys.path.append('/home/buyun/Documents/GitHub/NCVX')
    from ncvx import ncvx
    from ncvxStruct import Options, GeneralStruct

    device = torch.device( 'cuda')
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
    print("test 3/3 passed")



if __name__ == "__main__" :
    count = 1
    try:
        spectral_radius()
        count += 1
        dictionary_learning()
        count += 1
        sphere_manifold()
        print("Successfully passed all tests!")
    except Exception:
        print("Test {} fail, please carefully read the instructions on https://ncvx.org/".format(count))
    