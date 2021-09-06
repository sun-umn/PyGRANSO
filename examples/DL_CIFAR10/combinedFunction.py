from pygransoStruct import general_struct
import torch
import torch.nn as nn


def combinedFunctionDL(model,parameters = None):
    
    
    # objective function
    inputs = parameters.inputs
    labels = parameters.labels
    outputs = model(inputs)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    f = criterion(outputs, labels)


    # print("Exact Hessian Test: ")
    # model_cpu = model.to(device="cpu" )
    # labels_cpu = labels.to(device="cpu" )
    # inputs_cpu = inputs.to(device="cpu" )

    # model_cpu = model 
    # labels_cpu = labels 
    # inputs_cpu = inputs
    # obj_fn          = lambda x: criterion(model_cpu(x), labels_cpu)
    # H = torch.autograd.functional.hessian(obj_fn,inputs_cpu)

    # inequality constraint, matrix form
    ci = None

    # equality constraint 
    ce = None


    return [f,ci,ce]
