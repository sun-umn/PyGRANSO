import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct 
from pygranso.private.getNvar import getNvarTorch
from pygranso.private.getObjGrad import getObjGradDL
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from examples.model import new_rnn_modrelu
# from examples.model import rnn_modrelu


device = torch.device('cuda')

# sequence_length = 28
# input_size = 28
hidden_size = 30
# num_layers = 1
num_classes = 10
batch_size = 128
# num_epochs = 2
# learning_rate = 0.01

# double_precision = torch.float
double_precision = torch.double


class Model(nn.Module):
    def __init__(self, hidden_size):
        super(Model, self).__init__()
        self.rnn = new_rnn_modrelu.RNN(1, hidden_size)
        self.lin = nn.Linear(hidden_size, num_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs):

        state = self.rnn.default_hidden(inputs[:, 0, ...])
        for input in torch.unbind(inputs, dim=1):
            out_rnn, state = self.rnn(input.unsqueeze(dim=1), state)
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


# partial AD
def user_fn(model,inputs,labels):
    # objective function    
    logits = model(inputs)
    criterion = nn.CrossEntropyLoss()
    f = criterion(logits, labels)

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # A = list(model.parameters())[1]

    # inequality constraint
    ci = None

    # equality constraint 
    # special orthogonal group
    
    ce = None

    # ce = None

    return [f,ci,ce]

comb_fn = lambda model : user_fn(model,inputs,labels)


opts = pygransoStruct()
opts.torch_device = device
nvar = getNvarTorch(model.parameters())
opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
opts.opt_tol = 1e-3
opts.maxit = 10000
# opts.fvalquit = 1e-6
opts.print_level = 1
opts.print_frequency = 1
# opts.print_ascii = True
# opts.limited_mem_size = 100

opts.double_precision = True

# opts.globalAD = False # disable global auto-differentiation



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