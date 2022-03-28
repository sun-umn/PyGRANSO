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



device = torch.device('cuda')

sequence_length = 28*28
input_size = 1

# sequence_length = 28
# input_size = 28

hidden_size = 30

# hidden_size = 200

num_layers = 1
num_classes = 10
batch_size = 100

double_precision = torch.double

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
        x = torch.reshape(x,(batch_size,sequence_length,input_size))
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device, dtype=double_precision)
        out, hidden = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out[:, -1, :])
        return out

torch.manual_seed(5)

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device=device, dtype=double_precision)
model.train()

nn.init.orthogonal_(list(model.parameters())[1])
# nn.init.normal_(list(model.parameters())[1])

train_data = datasets.MNIST(
    root = '/home/buyun/Documents/GitHub/PyGRANSO/examples/data/mnist',
    train = True,
    transform = ToTensor(),
    download = True,
)

loaders = {
    'train' : torch.utils.data.DataLoader(train_data,
                                        batch_size=100,
                                        shuffle=True,
                                        num_workers=1),
}

inputs, labels = next(iter(loaders['train']))
inputs, labels = inputs.reshape(-1, sequence_length, input_size).to(device=device, dtype=double_precision), labels.to(device=device)

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def user_fn(model,inputs,labels):
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

    # ce.c1 = A.T @ A - torch.eye(hidden_size).to(device=device, dtype=double_precision)
    # ce.c2 = torch.det(A) - 1

    ce.c1 = norm(A.T @ A - torch.eye(hidden_size).to(device=device, dtype=double_precision),float('inf'))

    # ce = None

    return [f,ci,ce]

# # partial AD
# def user_fn(model,inputs,labels):
#     # objective function
#     logits = model(inputs)
#     criterion = nn.CrossEntropyLoss()
#     f = criterion(logits, labels)

#     A = list(model.parameters())[1]

#     # get f_grad by AD
#     n = getNvarTorch(model.parameters())
#     f_grad = getObjGradDL(nvar=n,model=model,f=f, torch_device=device, double_precision=True)
#     f = f.detach().item()

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

#     ce_grad[input_size*hidden_size:input_size*hidden_size+ hidden_size*(hidden_size),:] = M
#     ce_grad = ce_grad.detach()

#     return [f,f_grad,ci,ci_grad,ce,ce_grad]


comb_fn = lambda model : user_fn(model,inputs,labels)

opts = pygransoStruct()
opts.torch_device = device
nvar = getNvarTorch(model.parameters())
opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
# torch.manual_seed(1)
# opts.x0 = torch.randn(nvar,1).to(device=device, dtype=torch.double)
opts.opt_tol = 1e-6
opts.viol_eq_tol = 1e-5
opts.maxit = 2000
# opts.fvalquit = 1e-6
opts.print_level = 1
opts.print_frequency = 50
# opts.print_ascii = True
# opts.limited_mem_size = 100
opts.double_precision = True

# opts.steering_c_viol = 0.02
opts.mu0 = 100

opts.steering_c_mu = 0.95

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