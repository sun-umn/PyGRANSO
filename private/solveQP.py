import torch
from dbg_print import dbg_print, dbg_print_2
# import gurobipy as gp
# from gurobipy import GRB
import osqp
import numpy as np
from scipy import sparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

QP_REQUESTS = 0


def getErr():
    dbg_print("TODO: getErr")
    global QP_REQUESTS
    errors = 0
    return [QP_REQUESTS,errors]

def solveQP(H,f,A,b,LB,UB,QPsolver):
    

    # persistent requests;
    # persistent errors;

    # if isempty(requests)
    #     requests    = 0;
    #     errors      = 0;
    # end
    
    # if nargin < 4 && nargin > 0 && strcmpi(varargin{1},'counts')
    #     varargout   = {requests,errors};
    #     return
    # end
    
    # Todo: update requests
    global QP_REQUESTS
    QP_REQUESTS += 1

    try:
        if QPsolver == "osqp":
            # H,f always exist
            nvar = len(f)
            # H and A has to be sparse
            dbg_print_2("NOTE: solveQP.py inefficient convert torch tensor to numpy to sparse")
            H = H.cpu().numpy()
            f = f.cpu().numpy()
            if A != None:
                A = A.cpu().numpy()
            # b = b.cpu().numpy()
            LB = LB.cpu().numpy()
            UB = UB.cpu().numpy()
            H_sparse = sparse.csc_matrix(H)
            # LB and UB always exist

            if np.any(A != None) and np.any(b != None):
                Aeq = A
                beq = b
                speye = sparse.eye(nvar)
                LB_new = np.vstack((beq,LB))
                UB_new = np.vstack((beq,UB))
                A_new = sparse.vstack([Aeq,speye])
            else:
                #  no constraint A*x == b
                A_new = sparse.eye(nvar)
                LB_new = LB
                UB_new = UB

            # Create an OSQP object
            prob = osqp.OSQP()

            # Setup workspace and change alpha parameter
            prob.setup(H_sparse, f, A_new, LB_new, UB_new, alpha=1.0,verbose=False)

            # Solve problem
            res = prob.solve()

            solution = res.x
            sol_len = solution.size
            solution = solution.reshape((sol_len,1))
            solution = torch.from_numpy(solution).to(device=device, dtype=torch.double) 
            return solution

    except Exception as e:
        print(e)
