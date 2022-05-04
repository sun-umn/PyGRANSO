import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor

from pygranso.private.getObjGrad import getObjGradDL

from torch.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes,sequence_length,device,double_precision):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.device = device
        self.input_size = input_size
        self.double_precision = double_precision
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.reshape(x,(-1,self.sequence_length,self.input_size))
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=self.device, dtype=self.double_precision)
        out, hidden = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out[:, -1, :])
        return out


def get_model(rng_seed,input_size, hidden_size, num_layers, num_classes,device,double_precision,sequence_length):
    torch.manual_seed(rng_seed)
    model = RNN(input_size, hidden_size, num_layers, num_classes,sequence_length,device,double_precision).to(device=device, dtype=double_precision)
    model.train()
    # feasible init
    nn.init.orthogonal_(list(model.parameters())[1])
    return model

def get_data(sequence_length, input_size, device, double_precision,train_size,test_size):
    train_data = datasets.MNIST(
        root = '/home/buyun/Documents/GitHub/PyGRANSO/examples/data/mnist',
        train = True,
        transform = ToTensor(),
        download = True,
    )

    test_data = datasets.MNIST(
        root = '/home/buyun/Documents/GitHub/PyGRANSO/examples/data/mnist',
        train = False,
        transform = ToTensor(),
        download = True,
    )

    loaders = {
        'train' : torch.utils.data.DataLoader(train_data,
                                            batch_size=train_size,
                                            shuffle=True,
                                            num_workers=1),
        'test' : torch.utils.data.DataLoader(test_data,
                                            batch_size=test_size,
                                            shuffle=True,
                                            num_workers=1)
    }

    X_train, y_train = next(iter(loaders['train']))
    X_train, y_train = X_train.reshape(-1, sequence_length, input_size).to(device=device, dtype=double_precision), y_train.to(device=device)

    X_test, y_test = next(iter(loaders['test']))
    X_test, y_test = X_test.reshape(-1, sequence_length, input_size).to(device=device, dtype=double_precision), y_test.to(device=device)


    return X_train, y_train, X_test, y_test

def user_fn(model,inputs,labels,hidden_size,device,double_precision,maxfolding,unconstrained):
    # objective function
    logits = model(inputs)
    criterion = nn.CrossEntropyLoss()
    f = criterion(logits, labels)

    A = list(model.parameters())[1]

    # inequality constraint
    ci = None

    # equality constraint
    # special orthogonal group

    if not unconstrained:

        ce = pygransoStruct()

        ce.c1 = (A.T @ A - torch.eye(hidden_size).to(device=device, dtype=double_precision)).reshape(hidden_size**2,1)
        ce.c1 = torch.vstack((ce.c1,torch.det(A) - 1))

        if maxfolding == 'l1':
            ce.c1 = torch.sum(torch.abs(ce.c1))
        elif maxfolding == 'l2':
            ce.c1 = torch.sum(ce.c1**2)**0.5
        elif maxfolding == 'linf':
            ce.c1 = torch.amax(torch.abs(ce.c1))
        elif maxfolding == 'unfolding':
            ce.c1 = A.T @ A - torch.eye(hidden_size).to(device=device, dtype=double_precision)
        else:
            print("Please specficy you maxfolding type!")
            exit()
    
    else:
        ce = None

    return [f,ci,ce]

def get_opts(device,model,maxit,maxclocktime,double_precision,limited_mem_size):
    opts = pygransoStruct()
    opts.torch_device = device
    nvar = getNvarTorch(model.parameters())
    opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
    opts.opt_tol = 1e-6
    opts.viol_eq_tol = 1e-5
    opts.maxit = maxit
    # opts.fvalquit = 1e-6
    opts.print_level = 1
    opts.print_frequency = 10
    # opts.print_ascii = True
    # opts.limited_mem_size = 100
    if double_precision == torch.double:
        opts.double_precision = True
    else:
        opts.double_precision = False

    opts.maxclocktime = maxclocktime
    opts.limited_mem_size = limited_mem_size

    # opts.steering_c_viol = 0.02
    opts.mu0 = 100
    return opts

def get_model_acc(model,X,y,train, list):
    logits = model(X)
    _, predicted = torch.max(logits.data, 1)
    correct = (predicted == y).sum().item()
    acc = (100 * correct/len(X))

    if train:
        print("Initial train acc = {:.2f}%".format(acc))
    else:
        print("Initial test acc = {:.2f}%".format(acc))
    
    list = np.append(list,acc)
    return list

def get_restart_opts(device,model,maxit,maxclocktime,double_precision,soln,limited_mem_size,unconstrained):
    opts = pygransoStruct()
    opts.torch_device = device
    opts.opt_tol = 1e-6
    opts.viol_eq_tol = 1e-5
    opts.maxit = maxit
    # opts.fvalquit = 1e-6
    opts.print_level = 1
    opts.print_frequency = 10
    # opts.print_ascii = True
    # opts.limited_mem_size = 100
    if double_precision == torch.double:
        opts.double_precision = True
    else:
        opts.double_precision = False

    opts.maxclocktime = maxclocktime
    opts.limited_mem_size = limited_mem_size

    opts.x0 = soln.final.x
    if not unconstrained:
        opts.mu0 = soln.final.mu

    if limited_mem_size != 0:
        opts.limited_mem_warm_start = soln.H_final
        opts.scaleH0 = False
        opts.limited_mem_fixed_scaling = False
    else:
        opts.H0 = soln.H_final 
        opts.scaleH0 = False

    return opts

def make_plot(train_acc_list,test_acc_list,restart_time,maxit,title,save_plot):
    x_list = np.arange(0,maxit*(restart_time+2),maxit)
    plt.plot(x_list,train_acc_list,label='train acc')
    plt.plot(x_list,test_acc_list,label='test acc')
    plt.legend()
    plt.xlabel('pygranso iterations')
    plt.ylabel('accuracy (%)')
    plt.title(title)
    if not save_plot:
        plt.show()
    else:
        now = datetime.now() # current date and time
        date_time = now.strftime("%m%d%Y_%H:%M:%S")
        my_path = os.path.dirname(os.path.abspath(__file__))
        png_name = 'png/' + title + date_time
        plt.savefig(os.path.join(my_path, png_name))
        plt.clf()

def get_title(row_by_row,unconstrained,maxfolding,train_size,test_size,rng_seed,hidden_size):
    if row_by_row:
        sequence_type = 'row_by_row'
    else:
        sequence_type = 'pixel_by_pixel'

    if unconstrained:
        constr_name = 'unconstr'
    else:
        constr_name = maxfolding

    title = sequence_type +'_' + constr_name + "_trn{}_tst{}_seed{}_hidden_size{}".format(train_size,test_size,rng_seed,hidden_size)

    return title

def get_logname(title):
    now = datetime.now() # current date and time
    date_time = now.strftime("%m%d%Y_%H:%M:%S")
    my_path = os.path.dirname(os.path.abspath(__file__))
    logname = 'log/' + title + date_time
    return my_path,logname