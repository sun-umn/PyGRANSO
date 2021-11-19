import torch

def getCiVec(ci,torch_device):
    """
    getCiVec transforms the original tensor form constrained function into vector form
    """
    #  number of constraints
    nconstr = 0
    # get # of constraints
    # current constraint, e.g., c1, c2
    for constr_i in ci.__dict__.keys():
        constrMatrix = getattr(ci,constr_i)
        nconstr = nconstr + torch.numel(constrMatrix)

    # inquality constraints
    ci_vec_torch = torch.zeros((nconstr,1),device=torch_device, dtype=torch.double)
    curIdx = 0
    # nconstr_ci = genral_struct()
    nconstr_ci_total = 0
    # current constraint, e.g., c1, c2
    for constr_i in ci.__dict__.keys():
        constrMatrix = getattr(ci,constr_i)
        ci_vec_torch[curIdx:curIdx + torch.numel(constrMatrix)] = torch.reshape(constrMatrix,(torch.numel(constrMatrix),1))
        curIdx = curIdx + torch.numel(constrMatrix)
        # setattr(nconstr_ci,constr_i,torch.numel(constrMatrix))
        nconstr_ci_total += torch.numel(constrMatrix)

    ci_vec = ci_vec_torch.detach() # detach from current computational graph

    return [ci_vec,ci_vec_torch,nconstr_ci_total]