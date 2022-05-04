import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from perceptual_advex.distances import normalize_flatten_features
from torchvision import transforms
from torchvision import datasets
import torch.nn as nn
from torchvision.models import resnet50
import os
import numpy as np
import gc
import matplotlib.pyplot as plt
from datetime import datetime

class ResNet_orig_LPIPS(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        pretrained = bool(pretrained)
        print("Use pytorch pretrained weights: [{}]".format(pretrained))
        self.back = resnet50(pretrained=pretrained)
        self.back.fc = nn.Linear(2048, num_classes)
        # ===== Truncate the back and append the model to enable attack models
        model_list = list(self.back.children())
        self.head = nn.Sequential(
            *model_list[0:4]
        )
        self.layer1 = model_list[4]
        self.layer2 = model_list[5]
        self.layer3 = model_list[6]
        self.layer4 = model_list[7]
        self.tail = nn.Sequential(
            *[model_list[8],
              nn.Flatten(),
              model_list[9]]
            )    

    def features(self, x):
        """
            This function is called to produce perceptual features.
            Output ==> has to be a tuple of conv features.
        """
        x = x.type(self.back.fc.weight.dtype)
        x = self.head(x)
        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)
        return x_layer1, x_layer2, x_layer3, x_layer4
    
    def classifier(self, last_layer):
        last_layer = self.tail(last_layer)
        return last_layer
    
    def forward(self, x):
        return self.classifier(self.features(x)[-1])
    
    def features_logits(self, x):
        features = self.features(x)
        logits = self.classifier(features[-1])
        return features, logits

def load_base_model(device,precision):
    base_model = ResNet_orig_LPIPS(num_classes=100,pretrained=False).to(device=device, dtype = precision)
    # please download the checkpoint.pth from our Google Drive
    pretrained_path = os.path.join("/home/buyun/Documents/GitHub/PyGRANSO/examples/data/checkpoints/","checkpoint.pth")
    state_dict = torch.load(pretrained_path)["model_state_dict"]
    base_model.load_state_dict(state_dict)
    return base_model

def get_val_loader():
    # The ImageNet dataset is no longer publicly accessible. 
    # You need to download the archives externally and place them in the root directory
    valset = datasets.ImageNet('/home/buyun/Documents/datasets/ImageNet/', split='val', transform=transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor()]), download=False)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=1,shuffle=False, num_workers=0, collate_fn=None, pin_memory=False,)
    return val_loader

def move_data_device(device,precision,inputs,labels):
    # All the user-provided data (vector/matrix/tensor) must be in torch tensor format.
    # As PyTorch tensor is single precision by default, one must explicitly set `dtype=torch.double`.
    # Also, please make sure the device of provided torch tensor is the same as opts.torch_device.
    inputs = inputs.to(device=device, dtype=precision)
    labels = labels.to(device=device)
    return inputs,labels

def user_fn(X_struct, inputs, labels, lpips_model, model, attack_type, eps,box_constr):
    adv_inputs = X_struct.x_tilde + 1e-5
    epsilon = eps
    logits_outputs = model(adv_inputs)
    f = -torch.nn.functional.cross_entropy(logits_outputs,labels)

    # inequality constraint
    ci = pygransoStruct()
    constr_vec = (inputs-adv_inputs).reshape((inputs.numel()))
    if attack_type == 'L_2':
        ci.c1 = torch.linalg.norm(constr_vec,2) - epsilon
    elif attack_type == 'L_inf':
        ci.c1 = torch.amax(torch.abs(constr_vec)) - epsilon
    else:
        input_features = normalize_flatten_features( lpips_model.features(inputs)).detach()
        adv_features = lpips_model.features(adv_inputs)
        adv_features = normalize_flatten_features(adv_features)
        lpips_dists = (adv_features - input_features).norm(dim=1)
        ci.c1 = lpips_dists - epsilon

    if box_constr:
        box_constr_vec = torch.hstack((adv_inputs.reshape(inputs.numel())-1,-adv_inputs.reshape(inputs.numel())))
        ci.c2 = torch.sum(torch.clamp(box_constr_vec, min=0)**2)**0.5 # l2

    # equality constraint
    ce = None
    return [f,ci,ce]

def get_opts(device,maxit,opt_tol,viol_ineq_tol,print_frequency,limited_mem_size,mu0,inputs,precision,maxclocktime):
    opts = pygransoStruct()
    opts.print_use_orange = False
    opts.print_ascii = True
    opts.quadprog_info_msg  = False
    opts.torch_device = device
    opts.maxit = maxit
    opts.opt_tol = opt_tol 
    opts.viol_ineq_tol = viol_ineq_tol
    opts.print_frequency = print_frequency
    opts.limited_mem_size = limited_mem_size
    opts.mu0 = mu0
    opts.maxclocktime = maxclocktime
    opts.x0 = torch.reshape(inputs,(torch.numel(inputs),1)) + 1e-5*torch.randn((torch.numel(inputs),1)).to(device=device, dtype=precision)
    # opts.steering_c_viol = 0.99 # may be useful for l1 attack
    # opts.steering_c_mu = 0.1
    return opts

def visualize_attack(soln,inputs):

    def rescale_array(array):
        ele_min, ele_max = np.amin(array), np.amax(array)
        array = (array - ele_min) / (ele_max - ele_min)
        return array

    def tensor2img(tensor):
        tensor = torch.nn.functional.interpolate(
            tensor,
            scale_factor=3,
            mode="bilinear"
        )
        array = tensor.detach().cpu().numpy()[0, :, :, :]
        array = np.transpose(array, [1, 2, 0])
        return array

    final_adv_input = torch.reshape(soln.final.x,inputs.shape)

    ori_image = rescale_array(tensor2img(inputs))
    adv_image = rescale_array(tensor2img(final_adv_input))

    # print( np.amax(np.abs(adv_image-ori_image).reshape(inputs.size()[0], -1)))

    f = plt.figure()
    f.add_subplot(1,2,1)
    plt.imshow(ori_image)
    plt.title('Original Image')
    plt.axis('off')
    f.add_subplot(1,2,2)
    plt.imshow(adv_image)
    plt.title('Adversarial Attacked Image')
    plt.axis('off')
    plt.show()

def get_name(batch_size,maxclocktime,attack_type,box_constr,rng_seed):
    if box_constr:
        box_constr_str = 'box_cstr'
    else:
        box_constr_str = ''

    # save file
    now = datetime.now() # current date and time
    date_time = now.strftime("%m%d%Y_%H:%M:%S")
    my_path = os.path.dirname(os.path.abspath(__file__))
    name_str = "batch_size{}_maxtime{}_attack{}_{}_seed{}".format(batch_size,maxclocktime,attack_type,box_constr_str,rng_seed)
    log_name = "log/" + date_time + name_str + '.txt'

    print( name_str + "start\n\n")
    return [my_path, log_name, date_time, name_str]

def result_dict_init():

    result_dict = {
                'time':np.array([]),
                'iter': np.array([]),
                'F': np.array([]),
                'MF': np.array([]),
                'term_code_pass': np.array([]),
                'tv': np.array([]),
                'MF_tv': np.array([]),
                'fn_evals': np.array([]),
                'term_code_fail': [],
                'index_sort': np.array([])
                }

    return result_dict

def store_result(soln,end,start,img_idx,result_dict):
    if soln.termination_code != 12 and soln.termination_code != 8:

        result_dict['time'] = np.append(result_dict['time'],end-start)
        result_dict['F'] = np.append(result_dict['F'],soln.final.f)
        result_dict['MF'] = np.append(result_dict['MF'],soln.most_feasible.f)
        result_dict['term_code_pass'] = np.append(result_dict['term_code_pass'],soln.termination_code)
        result_dict['tv'] = np.append(result_dict['tv'],soln.final.tv) #total violation at x (vi + ve)
        result_dict['MF_tv'] = np.append(result_dict['MF_tv'],soln.most_feasible.tv)
        result_dict['iter'] = np.append(result_dict['iter'],soln.iters)
        result_dict['fn_evals'] = np.append(result_dict['fn_evals'],soln.fn_evals)
    else:
        result_dict['term_code_fail'].append("img_idx = {}, code = {}\n ".format(img_idx,soln.termination_code) )

    return result_dict

def sort_result(result_dict):

    index_sort = np.argsort(result_dict['F'])
    index_sort = index_sort[::-1]
    result_dict['F'] = result_dict['F'][index_sort]
    result_dict['time'] = result_dict['time'][index_sort]
    result_dict['MF'] = result_dict['MF'][index_sort]
    result_dict['term_code_pass'] = result_dict['term_code_pass'][index_sort]
    result_dict['tv'] = result_dict['tv'][index_sort]
    result_dict['MF_tv'] = result_dict['MF_tv'][index_sort]
    result_dict['iter'] = result_dict['iter'][index_sort]
    result_dict['fn_evals'] = result_dict['fn_evals'][index_sort]
    result_dict['index_sort'] = index_sort

def print_result(result_dict,total_img):

    print("Time = {}".format(result_dict['time']) )
    print("F obj = {}".format(result_dict['F']))
    print("MF obj = {}".format(result_dict['MF']))
    print("termination code = {}".format(result_dict['term_code_pass']))
    print("total violation tvi + tve = {}".format(result_dict['tv']))
    print("MF total violation tvi + tve = {}".format(result_dict['MF_tv']))
    print('iterations = {}'.format(result_dict['iter']))
    print("index sort = {}".format(result_dict['index_sort']))
    print("fn evals = {}".format(result_dict['fn_evals']))
    print("failed code: {}".format(result_dict['term_code_fail']))

    arr_len = result_dict['F'].shape[0]
    print("successful rate = {}".format(arr_len/total_img))

def get_data_name(my_path, date_time, name_str):
    data_name =  'data/' + date_time + name_str +'.npy'
    data_name = os.path.join(my_path, data_name)
    return data_name