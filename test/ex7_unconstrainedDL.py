#!/usr/bin/env python
# coding: utf-8

# # Unconstrained Deep Learning  
# 
# Train unconstrained deep learning for CIFAR-10 classification using modified LeNet5 based on [this PyTorch tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

# ## Problem Description

# We have a simple feed-forward network. The input is an image, which is fed through several layers to obtain the output. The logit output is used to decide the label of the input image. Below is a demo image of LeNet5:
# ![image.png](attachment:image.png)

# ## Modules Importing
# Import all necessary modules and add PyGRANSO src folder to system path. 

# In[1]:


import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso import pygranso
from pygransoStruct import pygransoStruct 
from private.getNvar import getNvarTorch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision


# ## Data Initialization 
# Specify torch device, neural network architecture, and generate data.
# 
# NOTE: please specify path for downloading data.
# 
# Use GPU for this problem. If no cuda device available, please set *device = torch.device('cpu')*

# In[2]:


device = torch.device('cuda')

class Net(nn.Module):
     def __init__(self):
             super().__init__()
             self.conv1 = nn.Conv2d(3, 6, 5)
             self.conv1_bn = nn.BatchNorm2d(6)
             self.pool = nn.MaxPool2d(2, 2)
             self.conv2 = nn.Conv2d(6, 8, 9)
             self.conv2_bn = nn.BatchNorm2d(8)
             self.fc1 = nn.Linear(8 * 3 * 3, 30)
             self.fc1_bn = nn.BatchNorm1d(30)
             self.fc2 = nn.Linear(30, 20)
             self.fc2_bn = nn.BatchNorm1d(20)
             self.fc3 = nn.Linear(20, 10)

     def forward(self, x):
             x = self.pool(F.elu( self.conv1_bn(self.conv1(x))  ))
             x = self.pool(F.elu( self.conv2_bn(self.conv2(x))  ))
             x = torch.flatten(x, 1) # flatten all dimensions except batch
             x = F.elu( self.fc1_bn(self.fc1(x)) )
             x = F.elu( self.fc2_bn(self.fc2(x)) )
             x = self.fc3(x)
             return x

# fix model parameters
torch.manual_seed(0)
model = Net().to(device=device, dtype=torch.double)

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 1000
trainset = torchvision.datasets.CIFAR10(root='/home/buyun/Documents/GitHub/PyGRANSO/examples', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
# data_in
for i, data in enumerate(trainloader, 0):        
    if i >= 1:
         break   
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    
# All the user-provided data (vector/matrix/tensor) must be in torch tensor format. 
# As PyTorch tensor is single precision by default, one must explicitly set `dtype=torch.double`.
# Also, please make sure the device of provided torch tensor is the same as opts.torch_device.
labels = labels.to(device=device) # label/target [256]
inputs = inputs.to(device=device, dtype=torch.double) # input data [256,3,32,32]


# ## Function Set-Up
# 
# Encode the optimization variables, and objective and constraint functions.
# 
# Note: please strictly follow the format of comb_fn, which will be used in the PyGRANSO main algortihm.

# In[3]:


# variables and corresponding dimensions.
var_in = {}
var_count = 0
var_str = "x"
for i in model.parameters():
    # print(i.shape)
    var_in[var_str+str(var_count)]= list(i.shape)
    var_count += 1
    
def user_fn(X_struct,model,inputs,labels):
    # objective function
    var_str = "x"
    var_count = 0
    for p in model.parameters():
        tmpstr = var_str+str(var_count)
        tmp_parameter = getattr(X_struct,tmpstr)
        p.data = tmp_parameter # update model parameters
        # p = tmp_parameter
        var_count += 1
    
    outputs = model(inputs)
    criterion = nn.CrossEntropyLoss()
    f = criterion(outputs, labels)
    ci = None
    ce = None
    return [f,ci,ce]

comb_fn = lambda X_struct : user_fn(X_struct,model,inputs,labels)


# ## User Options
# Specify user-defined options for PyGRANSO 

# In[4]:


opts = pygransoStruct()
opts.torch_device = device
nvar = getNvarTorch(model.parameters())
opts.maxit = 10000
opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
opts.opt_tol = 1e-6
opts.fvalquit = 1e-6
opts.print_level = 1
opts.print_frequency = 10
# opts.print_ascii = True


# opts.halt_on_linesearch_bracket = False
# opts.max_fallback_level = 3
# opts.min_fallback_level = 2
# opts.init_step_size = 1e-2
# opts.linesearch_maxit = 25
# opts.is_backtrack_linesearch = True
# opts.search_direction_rescaling = True
# opts.disable_terminationcode_6 = True

# BFGS may accumalet useless info
# only use most recent
# opts.limited_mem_size = 100



# ## Initial Test 
# Check initial accuracy of the modified LeNet5 model

# In[5]:


outputs = model(inputs )
acc = (outputs.max(1)[1] == labels).sum().item()/labels.size(0)

print("Initial acc = {}".format(acc)) 


# ## Main Algorithm

# In[6]:


# Main algorithm with logging enabled.
soln = pygranso(var_spec= [var_in,model], combined_fn = comb_fn, user_opts = opts)



# ## Train Accuracy

# In[7]:


torch.nn.utils.vector_to_parameters(soln.final.x, model.parameters())
outputs = model(inputs)
acc = (outputs.max(1)[1] == labels).sum().item()/labels.size(0)
print("Train acc = {}".format(acc))




# ## Test Accuracy

# In[8]:


# testset = torchvision.datasets.CIFAR10(root='/home/buyun/Documents/GitHub/PyGRANSO/examples', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=True, num_workers=2)

# for i, data in enumerate(testloader, 0):        
#     if i >= 3:
#          break   
#     # get the inputs; data is a list of [inputs, labels]
#     test_inputs, test_labels = data
    
# test_labels = test_labels.to(device=device ) # label/target [256]
# test_inputs = test_inputs.to(device=device, dtype=torch.double) # input data [256,3,32,32]

# test_outputs = model(test_inputs)
# test_acc = (test_outputs.max(1)[1] == test_labels).sum().item()/test_labels.size(0)
# print("Test acc = {}".format(test_acc))


# from makePlot import plot
# plot(log)