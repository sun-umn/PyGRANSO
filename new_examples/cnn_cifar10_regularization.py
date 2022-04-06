import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
import numpy as np

device = torch.device('cuda')

class Net(nn.Module):
     def __init__(self):
        super().__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=0)

     def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# fix model parameters
torch.manual_seed(2022)
model = Net().to(device=device, dtype=torch.double)

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 11
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
labels_one_hot = F.one_hot(labels, num_classes=10)

def user_fn(model,inputs,labels_one_hot):
    # objective function
    softmax_outputs = model(inputs)

    # criterion = nn.CrossEntropyLoss()
    # f = criterion(outputs, labels)

    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    w1 = list(model.parameters())[0] # conv1 weight
    w2 = list(model.parameters())[2] # conv2 weight

    f = torch.sum(w1**2) + torch.sum(w2**2)  

    ci = None
    ce = pygransoStruct()
    ce.c1 = torch.norm(softmax_outputs - labels_one_hot,1)

    # ce.c1 = torch.norm(softmax_outputs - labels_one_hot,float('inf'))
    

    return [f,ci,ce]

comb_fn = lambda model : user_fn(model,inputs,labels_one_hot)

opts = pygransoStruct()
opts.torch_device = device
nvar = getNvarTorch(model.parameters())
opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)

opts.opt_tol = 1e-4*np.sqrt(nvar)
# opts.fvalquit = 1e-6
opts.print_level = 1
opts.print_frequency = 50
opts.maxit = 600
# opts.print_ascii = True
opts.limited_mem_size = 100


outputs = model(inputs )
acc = (outputs.max(1)[1] == labels).sum().item()/labels.size(0)
# print(outputs.max(1)[1])
print("Initial acc = {}".format(acc))

start = time.time()
soln = pygranso(var_spec= model, combined_fn = comb_fn, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))

torch.nn.utils.vector_to_parameters(soln.final.x, model.parameters())
outputs = model(inputs)
acc = (outputs.max(1)[1] == labels).sum().item()/labels.size(0)
print("Train acc = {}".format(acc))