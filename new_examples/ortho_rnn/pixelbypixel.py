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
from torchvision import transforms
import numpy as np

from pygranso.private.getObjGrad import getObjGradDL



from torch.nn import Module
from torch.nn.parameter import Parameter
import sys
sys.path.append('/home/buyun/Documents/GitHub/unused/expRNN')
from parametrization import Parametrization
from orthogonal import Orthogonal
from initialization import cayley_init_
from trivializations import expm


class modrelu(nn.Module):
    def __init__(self, features):
        # For now we just support square layers
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)

        return phase * magnitude

exprnn = False

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if exprnn:
            mode = ("dynamic", 100, 100)
            param = expm
            self.recurrent_kernel = Orthogonal(hidden_size, hidden_size, initializer_skew = cayley_init_, mode = mode, param=param)
        else:
            self.recurrent_kernel = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        
        self.input_kernel = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, bias=False)
        
        self.nonlinearity = modrelu(hidden_size)
        # self.nonlinearity = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.input_kernel.weight.data, nonlinearity="relu")

    def default_hidden(self, input):
        return input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

    def forward(self, input, hidden):
        input = self.input_kernel(input)
        hidden = self.recurrent_kernel(hidden)
        out = input + hidden
        out = self.nonlinearity(out)

        return out, out



device = torch.device('cuda')

# sequence_length = 28
# input_size = 28
hidden_size = 30
# num_layers = 1
num_classes = 10
batch_size = 100
# num_epochs = 2
# learning_rate = 0.01

pixel_by_pixel = False

double_precision = torch.double

class Model(nn.Module):
    def __init__(self, hidden_size):
        super(Model, self).__init__()
        permute = False
        self.permute = permute
        permute = np.random.RandomState(92916)
        self.register_buffer("permutation", torch.LongTensor(permute.permutation(784)))

        if pixel_by_pixel:
            self.rnn = RNN(1, hidden_size)
        else:
            self.rnn = RNN(28, hidden_size)

        self.lin = nn.Linear(hidden_size, num_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs):
        if self.permute:
            inputs = inputs[:, self.permutation]
            
        state = self.rnn.default_hidden(inputs[:, 0, ...])

        if pixel_by_pixel:
            # for input in torch.unbind(inputs, dim=1):
            #     out_rnn, state = self.rnn(input.unsqueeze(dim=1), state)
            for i in range(28*28):
                out_rnn, state = self.rnn(inputs[:,i:(i+1)], state)
        else:
            for i in range(28):
                out_rnn, state = self.rnn(inputs[:,i*28:(i+1)*28], state)
        return self.lin(state)

    def loss(self, logits, y):
        return self.loss_func(logits, y)

    def correct(self, logits, y):
        return torch.eq(torch.argmax(logits, dim=1), y).float().sum()

torch.manual_seed(0)

kwargs = {'num_workers': 1, 'pin_memory': True}
# subset of loader ####################
index = list(range(0,batch_size))
trainset = datasets.MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor())
train_set_small = torch.utils.data.Subset(trainset,index)

testset = datasets.MNIST('./mnist', train=False, transform=transforms.ToTensor())
test_set_small = torch.utils.data.Subset(testset,index)

train_loader = torch.utils.data.DataLoader(
    train_set_small, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_set_small, batch_size=batch_size, shuffle=True, **kwargs)

########################################################

# Model and optimizers
model = Model(hidden_size).to(device=device, dtype=double_precision)
model.train()


inputs, labels = next(iter(train_loader))
inputs, labels = inputs.reshape(batch_size,784).to(device=device, dtype=double_precision), labels.to(device=device)

def user_fn(model,inputs,labels):
    # objective function    
    logits = model(inputs)
    criterion = nn.CrossEntropyLoss()
    f = criterion(logits, labels)

    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    A = list(model.parameters())[0]

    # inequality constraint
    ci = None

    # equality constraint 
    # special orthogonal group
    
    # ce = pygransoStruct()

    # ce.c1 = A.T @ A - torch.eye(hidden_size).to(device=device, dtype=double_precision)
    # ce.c2 = torch.det(A) - 1
    ce = None

    return [f,ci,ce]

# # partial AD
# def user_fn(model,inputs,labels):
#     # objective function    
#     logits = model(inputs)
#     criterion = nn.CrossEntropyLoss()
#     f = criterion(logits, labels)

#     # get f_grad by AD
#     n = getNvarTorch(model.parameters())
#     f_grad = getObjGradDL(nvar=n,model=model,f=f, torch_device=device, double_precision=True)
#     f = f.detach().item()

#     # for param_tensor in model.state_dict():
#     #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

#     A = list(model.parameters())[0]

#     # inequality constraint
#     ci = None
#     ci_grad = None

#     # equality constraint 
#     # special orthogonal group
    
#     ce = pygransoStruct()

#     ce = A.T @ A - torch.eye(hidden_size).to(device=device, dtype=double_precision)
#     ce = ce.detach()
#     nconstr = hidden_size*hidden_size
#     ce = torch.reshape(ce,(nconstr,1))
#     ce_grad = torch.zeros((n,nconstr)).to(device=device, dtype=double_precision)
#     M = torch.zeros((nconstr,nconstr)).to(device=device, dtype=double_precision)

#     for i in range(hidden_size):
#         for j in range(hidden_size):
#             J_ij = torch.zeros((hidden_size,hidden_size)).to(device=device, dtype=double_precision)
#             J_ij[i,j] = 1
#             tmp = A.T@J_ij + J_ij.T@A
#             M[hidden_size*i+j,:] = tmp.reshape((1,hidden_size*hidden_size))

#     ce_grad[0:hidden_size*(hidden_size),:] = M
#     ce_grad = ce_grad.detach()

#     return [f,f_grad,ci,ci_grad,ce,ce_grad]

comb_fn = lambda model : user_fn(model,inputs,labels)


opts = pygransoStruct()
opts.torch_device = device
nvar = getNvarTorch(model.parameters())
opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
opts.opt_tol = 5e-4
opts.viol_eq_tol = 1e-5
opts.maxit = 100
# opts.fvalquit = 1e-6
opts.print_level = 1
opts.print_frequency = 1
# opts.print_ascii = True
# opts.limited_mem_size = 100
opts.double_precision = True
opts.limited_mem_size = 100



# opts.globalAD = False # disable global auto-differentiation


opts.mu0 = 200


logits = model(inputs)
_, predicted = torch.max(logits.data, 1)
correct = (predicted == labels).sum().item()
print("Initial acc = {:.2f}%".format((100 * correct/len(inputs))))  

start = time.time()
soln = pygranso(var_spec= model, combined_fn = comb_fn, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))

torch.nn.utils.vector_to_parameters(soln.final.x, model.parameters())
logits = model(inputs)
_, predicted = torch.max(logits.data, 1)
correct = (predicted == labels).sum().item()
print("Final acc = {:.2f}%".format((100 * correct/len(inputs))))  