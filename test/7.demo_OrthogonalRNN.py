#!/usr/bin/env python
# coding: utf-8

# # Orthogonal RNN
# 
# This notebook contains examples of how to train RNN with otrhogonal constraints.
# 
# Reference: Lezcano-Casado, Mario, and David Martınez-Rubio. "Cheap orthogonal constraints in neural networks: A simple parametrization of the orthogonal and unitary group." International Conference on Machine Learning. PMLR, 2019.

# ## Problem Description

# $$\min_{B\in G}f(B),$$
# where $f$ is our classifier implemented in RNN, $B$ is the recurrent kernel/matrix, and $G$ is the orthogonal group
# $$G=\{ B\in R^{n\times n} | B^TB=I, \det(B)=1 \}$$

# ## Modules Importing
# Import all necessary modules and add NCVX src folder to system path. NCVX src folder to system path.

# In[1]:


import time
import torch
import sys
## Adding NCVX directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/NCVX')
from ncvx import ncvx
from ncvxStruct import Options, GeneralStruct 
from private.getNvar import getNvarTorch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor


# ## Data Generation 
# Specify torch device, neural network architecture, and generate data.
# 
# NOTE: please specify path for downloading data

# In[2]:


device = torch.device('cuda')

sequence_length = 28
input_size = 28
hidden_size = 30
num_layers = 1
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        pass
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device, dtype=torch.double)
        out, hidden = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out[:, -1, :])
        return out
    
torch.manual_seed(0)

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device=device, dtype=torch.double)
model.train()

train_data = datasets.MNIST(
    root = '/home/buyun/Documents/GitHub/NCVX/examples/data/mnist',
    train = True,                         
    transform = ToTensor(), 
    download = False,            
)
test_data = datasets.MNIST(
    root = '/home/buyun/Documents/GitHub/NCVX/examples/data/mnist', 
    train = False, 
    transform = ToTensor()
)

loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                        batch_size=100, 
                                        shuffle=True, 
                                        num_workers=1),

    'test'  : torch.utils.data.DataLoader(test_data, 
                                        batch_size=100, 
                                        shuffle=True, 
                                        num_workers=1),
}

inputs, labels = next(iter(loaders['train']))
inputs, labels = inputs.reshape(-1, sequence_length, input_size).to(device=device, dtype=torch.double), labels.to(device=device)


# ## Problem Definition
# 
# Spceify torch device, optimization variables, and corresponding objective and constrained function.
# 
# Note: please strictly follow the format of evalObjFunction and combinedFunction, which will be used in the NCVX main algortihm. *X_struct* and *data_in* are always required.

# In[3]:


# variables and corresponding dimensions.
var_in = {}
var_count = 0
var_str = "x"
for i in model.parameters():
    # print(i.shape)
    var_in[var_str+str(var_count)]= list(i.shape)
    var_count += 1

def obj_eval_fn(X_struct):
    var_str = "x"
    var_count = 0
    for p in model.parameters():
        tmpstr = var_str+str(var_count)
        tmp_parameter = getattr(X_struct,tmpstr)
        tmp_parameter.requires_grad_(True)
        p.data = tmp_parameter
        var_count += 1

    logits = model(inputs)
    criterion = nn.CrossEntropyLoss()
    f = criterion(logits, labels)
    return f

def comb_fn(X_struct):
    var_str = "x"
    var_count = 0
    device = torch.device('cuda')

    for p in model.parameters():
        tmpstr = var_str+str(var_count)
        tmp_parameter = getattr(X_struct,tmpstr)
        # Obtain the recurrent parameter with dimension n by n, where n is the number of features in the hidden state h
        if tmp_parameter.shape == torch.Size([hidden_size, hidden_size]):
            A = tmp_parameter
        tmp_parameter.requires_grad_(True)
        p.data = tmp_parameter
        var_count += 1

    logits = model(inputs)
    criterion = nn.CrossEntropyLoss()
    f = criterion(logits, labels)
    # inequality constraint
    ci = None

    # equality constraint 

    # special orthogonal group
    ce = GeneralStruct()
    ce.c1 = A.T @ A - torch.eye(hidden_size).to(device=device, dtype=torch.double)
    ce.c2 = torch.det(A) - 1
    return [f,ci,ce]


# ## User Options
# Specify user-defined options for NCVX algorithm

# In[4]:


opts = Options()
nvar = getNvarTorch(model.parameters())
opts.QPsolver = 'osqp' 
opts.maxit = 100
opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
opts.opt_tol = 1e-6
opts.fvalquit = 1e-6
opts.print_level = 1
opts.print_frequency = 10
 # opts.max_fallback_level = 3
# opts.min_fallback_level = 2
# opts.init_step_size = 1e-2
opts.init_step_size = 1e-1
opts.halt_on_linesearch_bracket = False
# opts.disable_terminationcode_6 = True

opts.linesearch_maxit = 25
# opts.linesearch_maxit = 10
opts.is_backtrack_linesearch = True
opts.searching_direction_rescaling = True
# opts.limited_mem_size = 200


# ## Initial Test 
# Check initial accuracy of RNN model

# In[5]:


logits = model(inputs)
_, predicted = torch.max(logits.data, 1)
correct = (predicted == labels).sum().item()
print("Initial acc = {:.2f}%".format((100 * correct/len(inputs))))  


# ## Main Algorithm

# In[6]:


start = time.time()
soln = ncvx(combinedFunction = comb_fn, objEvalFunction = obj_eval_fn, var_dim_map = var_in, nn_model= model, torch_device = device, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))


# ## Train Accuracy

# In[7]:


torch.nn.utils.vector_to_parameters(soln.final.x, model.parameters())
logits = model(inputs)
_, predicted = torch.max(logits.data, 1)
correct = (predicted == labels).sum().item()
print("Final acc = {:.2f}%".format((100 * correct/len(inputs))))     

