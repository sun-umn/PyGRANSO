from pygransoStruct import general_struct
import torch
import torch.nn as nn


def combinedFunctionDL(model,parameters = None):
    
    
    # objective function
    inputs = parameters.inputs
    labels = parameters.labels
    outputs = model(inputs)
    criterion = nn.CrossEntropyLoss()
    f = criterion(outputs, labels)

    # inequality constraint, matrix form
    ci = None

    # equality constraint 
    ce = None


    return [f,ci,ce]
