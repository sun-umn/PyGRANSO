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
        # tmp_parameter.requires_grad_(True)
        p.data = tmp_parameter
        var_count += 1

    inputs = data_in.inputs
    labels = data_in.labels
    logits = model(inputs)
    # correct = model.correct(logits, labels)
    criterion = nn.CrossEntropyLoss()
    f = criterion(logits, labels)
    return f

def combinedFunction(X_struct, data_in = None):
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
    # criterion = nn.CrossEntropyLoss()
    f = model.loss(logits, labels)
    # inequality constraint
    ci = None
    # equality constraint 
    ce = None
    return [f,ci,ce]
