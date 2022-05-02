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

def user_fn(model,inputs,labels,hidden_size,device,double_precision,maxfolding):
    # objective function
    logits = model(inputs)
    criterion = nn.CrossEntropyLoss()
    f = criterion(logits, labels)

    A = list(model.parameters())[1]

    # inequality constraint
    ci = None

    # equality constraint
    # special orthogonal group

    ce = pygransoStruct()

    constr_vec = (A.T @ A - torch.eye(hidden_size).to(device=device, dtype=double_precision)).reshape(hidden_size**2,1)
    constr_vec = torch.vstack((constr_vec,torch.det(A) - 1))

    if maxfolding == 'l1':
        ce.c1 = torch.sum(torch.abs(constr_vec))
    elif maxfolding == 'l2':
        ce.c1 = torch.sum(constr_vec**2)**0.5
    elif maxfolding == 'linf':
        ce.c1 = torch.amax(torch.abs(constr_vec))
    elif maxfolding == 'unfolding':
        ce.c1 = A.T @ A - torch.eye(hidden_size).to(device=device, dtype=double_precision)
    else:
        print("Please specficy you maxfolding type!")
        exit()

    return [f,ci,ce]

def get_opts(device,model,maxit,maxclocktime,double_precision):
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
    # opts.steering_c_viol = 0.02
    opts.mu0 = 100
    return opts

def get_model_acc(model,X,y,train):
    logits = model(X)
    _, predicted = torch.max(logits.data, 1)
    correct = (predicted == y).sum().item()

    if train:
        print("Initial train acc = {:.2f}%".format((100 * correct/len(X))))
    else:
        print("Initial test acc = {:.2f}%".format((100 * correct/len(X))))