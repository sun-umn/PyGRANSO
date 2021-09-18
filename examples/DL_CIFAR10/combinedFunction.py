from pygransoStruct import general_struct
import torch
import torch.nn as nn

def eval_obj(model,data_in = None):
    # objective function
    inputs = data_in.inputs
    labels = data_in.labels
    outputs = model(inputs)
    criterion = nn.CrossEntropyLoss()
    f = criterion(outputs, labels)
    return f

def combinedFunction(model,data_in = None):
    # objective function
    inputs = data_in.inputs
    labels = data_in.labels
    outputs = model(inputs)
    criterion = nn.CrossEntropyLoss()
    f = criterion(outputs, labels)
    # inequality constraint
    ci = None
    # equality constraint 
    ce = None
    return [f,ci,ce]
