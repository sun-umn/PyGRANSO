from pygransoStruct import general_struct
import torch
import torch.nn as nn

def eval_obj(X_struct, data_in = None):
    # objective function
    # user defined variable, matirx form. torch tensor

    var_str = "x"
    model = data_in.model
    var_count = 0

    for p in model.parameters():
        tmpstr = var_str+str(var_count)
        tmp_parameter = getattr(X_struct,tmpstr)
        tmp_parameter.requires_grad_(True)
        p.data = tmp_parameter
        var_count += 1

    inputs = data_in.inputs
    labels = data_in.labels
    logits = model(inputs)
    criterion = nn.CrossEntropyLoss()
    f = criterion(logits, labels)
    return f

def combinedFunction(X_struct, data_in = None):
    # objective function
    # user defined variable, matirx form. torch tensor

    var_str = "x"
    model = data_in.model
    var_count = 0
    n = data_in.hidden_size
    device = torch.device('cuda')

    for p in model.parameters():
        tmpstr = var_str+str(var_count)
        tmp_parameter = getattr(X_struct,tmpstr)
        # Obtain the recurrent parameter with dimension n by n, where n is the number of features in the hidden state h
        if tmp_parameter.shape == torch.Size([n, n]):
            A = tmp_parameter
        tmp_parameter.requires_grad_(True)
        p.data = tmp_parameter
        var_count += 1

    inputs = data_in.inputs
    labels = data_in.labels
    logits = model(inputs)
    criterion = nn.CrossEntropyLoss()
    f = criterion(logits, labels)
    # inequality constraint
    ci = None

    # equality constraint 
    
    # # Unitary group
    # ce = general_struct()
    # ce.c1 = torch.conj(A).T @ A - torch.eye(n).to(device=device, dtype=torch.double)

    # special orthogonal group
    ce = general_struct()
    ce.c1 = A.T @ A - torch.eye(n).to(device=device, dtype=torch.double)
    ce.c2 = torch.det(A) - 1
    
    # ce = None

    return [f,ci,ce]
