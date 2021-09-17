from pygransoStruct import general_struct
import torch
import torch.nn as nn

def eval_obj(model,parameters = None):
    # objective function
    inputs = parameters.inputs
    labels = parameters.labels
    outputs = model(inputs)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    f = criterion(outputs, labels)
    return f

def combinedFunction(model,parameters = None):
    # objective function
    f = eval_obj(model,parameters)
    # inequality constraint
    ci = None
    # equality constraint 
    ce = None
    return [f,ci,ce]
