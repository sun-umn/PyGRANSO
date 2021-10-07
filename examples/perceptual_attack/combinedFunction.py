from pygransoStruct import general_struct
import torch

def MarginLoss(logits,labels):
    correct_logits = torch.gather(logits, 1, labels.view(-1, 1))
    max_2_logits, argmax_2_logits = torch.topk(logits, 2, dim=1)
    top_max, second_max = max_2_logits.chunk(2, dim=1)
    top_argmax, _ = argmax_2_logits.chunk(2, dim=1)
    labels_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
    labels_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
    max_incorrect_logits = labels_eq_max * second_max + labels_ne_max * top_max
    loss = -(max_incorrect_logits - correct_logits).clamp(max=1).squeeze().sum()
    return loss

def eval_obj(X_struct,data_in = None):
    # user defined variable, matirx form. torch tensor
    adv_input = X_struct.x_tilde
    adv_input.requires_grad_(True)
    
    # objective function
    epsilon = data_in.epsilon
    inputs = data_in.inputs
    labels = data_in.labels
    model = data_in.model
    logits_outputs = model(adv_input)

    f = MarginLoss(logits_outputs,labels)
    return f

def combinedFunction(X_struct, data_in = None):
    
    # user defined variable, matirx form. torch tensor
    adv_input = X_struct.x_tilde
    adv_input.requires_grad_(True)
    
    # objective function
    epsilon = data_in.epsilon
    inputs = data_in.inputs
    labels = data_in.labels
    model = data_in.model
    logits_outputs = model(adv_input)

    f = MarginLoss(logits_outputs,labels)

    # inequality constraint, matrix form
    ci = None
    # ci = general_struct()
    # ci.c1 = la.norm(phi(inputs) - phi(adv_input) - epsilon ) 

    # equality constraint 
    ce = None

    return [f,ci,ce]