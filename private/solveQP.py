import torch
import osqp
import numpy as np
from scipy import sparse
import traceback,sys

QP_REQUESTS = 0

def solveQP(H,f,A,b,LB,UB,QPsolver,torch_device, double_precision):
    """
    solveQP will call the QP solver to calculate the given QP
    """
    global QP_REQUESTS
    QP_REQUESTS += 1

    try:
        if QPsolver == "osqp":
            # H,f always exist
            nvar = len(f)
            # H and A has to be sparse
            H = H.cpu().numpy()
            # avoid numerical issue in ncvx
            # epsilon = 1e-6
            # H = H + epsilon * np.eye(nvar)
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
                A_new = sparse.csc_matrix(A_new)
            else:
                #  no constraint A*x == b
                A_new = sparse.eye(nvar)
                A_new = sparse.csc_matrix(A_new)
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
            if double_precision:
                torch_dtype = torch.double
            else:
                torch_dtype = torch.float
            solution = torch.from_numpy(solution).to(device=torch_device, dtype=torch_dtype) 
            return solution

    except Exception as e:
        [w,v] = np.linalg.eigh(H)
        w_sorted = np.sort(w)
        print(w_sorted)
        print(traceback.format_exc())
        sys.exit()


def getErr():
    # getErr NOT used
    global QP_REQUESTS
    errors = 0
    return [QP_REQUESTS,errors]