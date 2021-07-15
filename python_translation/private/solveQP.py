from dbg_print import dbg_print
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy import conjugate as conj

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

    if QPsolver == "gurobi":
        
        # H,f always exist
        # LB and UB always exist
        #  formulation of QP has no 1/2
        H = H/2

        nvar = len(f)
        # nvar = H.shape[0]
        # Create a new model
        m = gp.Model()
        vtype = [GRB.CONTINUOUS] * nvar
        
        # Add variables to model
        vars = []
        for j in range(nvar):
            vars.append(m.addVar(lb=LB[j], ub=UB[j], vtype=vtype[j]))
        x_vec = np.array(vars).reshape(nvar,1)

        if np.any(A != None) and np.any(b != None):
            Aeq = A
            beq = b
            # Populate A matrix
            expr = gp.LinExpr()
            Ax = Aeq @ x_vec
            expr += Ax[0,0]
            m.addLConstr(expr, GRB.GREATER_EQUAL, beq)
        else:
            #  no constraint A*x < b
            pass

        solution = np.zeros((nvar,1))

        # Populate objective: x.THx + f.T x
        obj = gp.QuadExpr()
        xTHx = x_vec.T @ H @ x_vec + f.T @ x_vec
        obj += xTHx[0,0]
        m.setObjective(obj)

        #  suppress output
        # m.Params.LogToConsole = 0
        m.Params.outputflag = 0

        m.optimize()
        x = m.getAttr('x', vars)
        for i in range(nvar):
            solution[i,0] = x[i]

        return solution
