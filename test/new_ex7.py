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

# variables and corresponding dimensions.
var_in = {}
var_count = 0
var_str = "x"
for i in model.parameters():
    # print(i.shape)
    var_in[var_str+str(var_count)]= list(i.shape)
    var_count += 1
    
# def user_fn(X_struct,model,inputs,labels):
#     # objective function
#     var_str = "x"
#     var_count = 0
#     for p in model.parameters():
#         tmpstr = var_str+str(var_count)
#         tmp_parameter = getattr(X_struct,tmpstr)
#         p.data = tmp_parameter # update model parameters
#         # p = tmp_parameter
#         var_count += 1
    
#     outputs = model(inputs)
#     criterion = nn.CrossEntropyLoss()
#     f = criterion(outputs, labels)
#     ci = None
#     ce = None
#     return [f,ci,ce]

def user_fn(X_struct,model,inputs,labels):
    # objective function
    var_str = "x"
    var_count = 0
    for p in model.parameters():
        tmpstr = var_str+str(var_count)
        tmp_parameter = getattr(X_struct,tmpstr)
        if tmp_parameter.shape == torch.Size([6]):
            A = tmp_parameter
        p.data = tmp_parameter # update model parameters
        # p = tmp_parameter
        var_count += 1
    
    outputs = model(inputs)
    criterion = nn.CrossEntropyLoss()
    f = criterion(outputs, labels)
    ci = pygransoStruct()
    ci.c1 = -A**2
    ce = None
    return [f,ci,ce]



comb_fn = lambda X_struct : user_fn(X_struct,model,inputs,labels)

opts = pygransoStruct()
opts.torch_device = device
nvar = getNvarTorch(model.parameters())
opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
opts.opt_tol = 1e-5
# opts.fvalquit = 1e-6
opts.print_level = 1
opts.print_frequency = 1
# opts.print_ascii = True

outputs = model(inputs )
acc = (outputs.max(1)[1] == labels).sum().item()/labels.size(0)

print("Initial acc = {}".format(acc)) 

start = time.time()
soln = pygranso(var_spec= [var_in,model], combined_fn = comb_fn, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))

torch.nn.utils.vector_to_parameters(soln.final.x, model.parameters()) # update model
outputs = model(inputs)
acc = (outputs.max(1)[1] == labels).sum().item()/labels.size(0)
print("Train acc = {}".format(acc))