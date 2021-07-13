import gurobipy as gp
import numpy as np


class Class_solveQP:
    def __init__(self):
        self.requests = 0


    def solveQP(self,H,f,A,b,LB,UB,QPsolver):
        
    
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
        self.requests += 1
        if QPsolver == "gurobi":
            nvar = len(f)
            # Create a new model
            m = gp.Model("QPsolver")

            # Create variables
            x = m.addVars(nvar,1)
            lst = []
            for i in range(len(x)):
                lst.append(x[i,0])
            x_vec = np.array(lst)
            x_vec = x_vec.reshape(nvar,1)

            #  formulation of QP has no 1/2
            H = H/2
            # H,f always exist
            
            # Set objective: x.THx + f.T x
            obj = x_vec.T @ H @ x_vec + f.T @ x_vec
            m.setObjective(obj[0][0])
            
            if A != None and b != None:
                Aeq = A
                beq = b
                m.addConstrs(Aeq @ x_vec == beq)
            else:
                #  no constraint A*x < b
                pass
            
            # LB and UB always exist
            m.addConstrs(x_vec[i][0] >= LB[i] for i in range(len(LB)))
            m.addConstrs(x_vec[i][0] <= UB[i] for i in range(len(UB)))
            
            #  suppress output
            # m.Params.LogToConsole = 0
            m.Params.outputflag = 0
            
            m.optimize()

            result_lst = []
            for v in m.getVars():
                result_lst.append(v.x)
            result  = np.array([result_lst]).T

        return result