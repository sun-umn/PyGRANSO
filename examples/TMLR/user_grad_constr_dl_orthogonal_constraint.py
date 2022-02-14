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
from examples.model import rnn_modrelu

device = torch.device('cuda')

sequence_length = 28
input_size = 28
hidden_size = 30
num_layers = 1
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# double_precision = torch.float
double_precision = torch.double


class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.rnn = rnn_modrelu.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device, dtype=double_precision)
        out, hidden = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out[:, -1, :])
        return out

torch.manual_seed(0)

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device=device, dtype=double_precision)
model.train()

train_data = datasets.MNIST(
    root = '/home/buyun/Documents/GitHub/PyGRANSO/examples/data/mnist',
    train = True,                         
    transform = transforms.ToTensor(), 
    download = True,            
)
# test_data = datasets.MNIST(
#     root = '/home/buyun/Documents/GitHub/PyGRANSO/examples/data/mnist', 
#     train = False, 
#     transform = ToTensor()
# )

loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        num_workers=1),

    # 'test'  : torch.utils.data.DataLoader(test_data, 
    #                                     batch_size=100, 
    #                                     shuffle=True, 
    #                                     num_workers=1),
}

inputs, labels = next(iter(loaders['train']))
inputs, labels = inputs.reshape(-1, sequence_length, input_size).to(device=device, dtype=double_precision), labels.to(device=device)

# # Load data (exprnn) #######################################
# kwargs = {'num_workers': 1, 'pin_memory': True}
# # train_loader = torch.utils.data.DataLoader(
# #     datasets.MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor()),
# #     batch_size=batch_size, shuffle=True, **kwargs)
# # test_loader = torch.utils.data.DataLoader(
# #     datasets.MNIST('./mnist', train=False, transform=transforms.ToTensor()),
# #     batch_size=batch_size, shuffle=True, **kwargs)

# # subset of loader ####################
# index = list(range(0,128))
# trainset = datasets.MNIST('/home/buyun/Documents/GitHub/PyGRANSO/examples/data/mnist', train=True, download=True, transform=transforms.ToTensor())
# train_set_small = torch.utils.data.Subset(trainset,index)

# testset = datasets.MNIST('/home/buyun/Documents/GitHub/PyGRANSO/examples/data/mnist', train=False, transform=transforms.ToTensor())
# test_set_small = torch.utils.data.Subset(testset,index)

# train_loader = torch.utils.data.DataLoader(
#     train_set_small, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     test_set_small, batch_size=batch_size, shuffle=True, **kwargs)

# inputs, labels = next(iter(train_loader))
# inputs, labels = inputs.reshape(-1, sequence_length, input_size).to(device=device, dtype=double_precision), labels.to(device=device)



# partial AD
def user_fn(model,inputs,labels):
    # objective function    
    logits = model(inputs)
    criterion = nn.CrossEntropyLoss()
    f = criterion(logits, labels)

    # get f_grad by AD
    n = getNvarTorch(model.parameters())
    f_grad = getObjGradDL(nvar=n,model=model,f=f, torch_device=device, double_precision=True)
    f = f.detach().item()

    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    A = list(model.parameters())[1]

    # inequality constraint
    ci = None
    ci_grad = None

    # equality constraint 
    # special orthogonal group
    
    ce = pygransoStruct()

    ce = A.T @ A - torch.eye(hidden_size).to(device=device, dtype=double_precision)
    ce = ce.detach()
    nconstr = hidden_size*hidden_size
    ce = torch.reshape(ce,(nconstr,1))
    ce_grad = torch.zeros((n,nconstr)).to(device=device, dtype=double_precision)
    M = torch.zeros((nconstr,nconstr)).to(device=device, dtype=double_precision)

    for i in range(hidden_size):
        for j in range(hidden_size):
            J_ij = torch.zeros((hidden_size,hidden_size)).to(device=device, dtype=double_precision)
            J_ij[i,j] = 1
            tmp = A.T@J_ij + J_ij.T@A
            M[hidden_size*i+j,:] = tmp.reshape((1,hidden_size*hidden_size))

    ce_grad[hidden_size*input_size:hidden_size*(input_size+hidden_size),:] = M
    ce_grad = ce_grad.detach()

    return [f,f_grad,ci,ci_grad,ce,ce_grad]

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

opts.globalAD = False # disable global auto-differentiation



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