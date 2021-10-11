from pygransoStruct import general_struct
from perceptual_advex.distances import normalize_flatten_features
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
    adv_inputs = X_struct.x_tilde
    adv_inputs.requires_grad_(True)
    
    # objective function
    labels = data_in.labels
    model = data_in.model
    logits_outputs = model(adv_inputs)

    f = MarginLoss(logits_outputs,labels)
    return f

def combinedFunction(X_struct, data_in = None):
    
    # user defined variable, matirx form. torch tensor
    adv_inputs = X_struct.x_tilde
    adv_inputs.requires_grad_(True)
    
    # objective function
    # 8/255 for L_inf, 1 for L_2, 0.5 for PPGD/LPA
    if data_in.attack_type == 'L_2':
        epsilon = 1
    elif data_in.attack_type == 'L_inf':
        epsilon = 8/255
    else:
        epsilon = 0.5

    inputs = data_in.inputs
    labels = data_in.labels
    model = data_in.model
    logits_outputs = model(adv_inputs)

    f = MarginLoss(logits_outputs,labels)

    # inequality constraint, matrix form
    # ci = None
    ci = general_struct()
    if data_in.attack_type == 'L_2':
        ci.c1 = torch.norm((inputs - adv_inputs).reshape(inputs.size()[0], -1)) - epsilon
    elif data_in.attack_type == 'L_inf':
        ci.c1 = torch.norm((inputs - adv_inputs).reshape(inputs.size()[0], -1), float('inf')) - epsilon
    else:
        lpips_model = data_in.lpips_model
        input_features = normalize_flatten_features( lpips_model.features(inputs)).detach()
        adv_features = lpips_model.features(adv_inputs)
        adv_features = normalize_flatten_features(adv_features)
        lpips_dists = (adv_features - input_features).norm(dim=1)
        ci.c1 = lpips_dists - epsilon
    
    # ci.c1 = la.norm(phi(inputs) - phi(adv_input) - epsilon ) 

    # equality constraint 
    ce = None

    return [f,ci,ce]