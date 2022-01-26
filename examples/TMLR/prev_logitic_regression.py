import time
import torch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn

device = torch.device('cuda')
batch_size = 100
m = batch_size
torch.manual_seed(0)

train_dataset = datasets.MNIST(
    root = '/home/buyun/Documents/GitHub/PyGRANSO/examples/data/mnist',
    train = True,                         
    transform = ToTensor(), 
    download = False,            
)
test_dataset = datasets.MNIST(
    root = '/home/buyun/Documents/GitHub/PyGRANSO/examples/data/mnist', 
    train = False, 
    transform = ToTensor()
)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

inputs, labels = next(iter(train_loader))
inputs, labels = inputs.reshape(-1, 28 * 28).to(device=device, dtype=torch.double), labels.to(device=device)

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim).to(device=device, dtype=torch.double)

    def forward(self, x):
        outputs = self.linear(x).to(device=device, dtype=torch.double)
        return outputs
    
input_dim = 784
output_dim = 10
model = LogisticRegression(input_dim, output_dim)

lambda_r = 0.01

def comb_fn(model):
    # objective function
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    w = list(model.parameters())[0]

    outputs = model(inputs)
    criterion = nn.CrossEntropyLoss()
    penalty = lambda_r*torch.norm(w,p=2)
    f = criterion(outputs, labels) 
    # ci = None
    ci = pygransoStruct()
    ci.c1 = penalty - 0.01
    ce = None
    return [f,ci,ce]



opts = pygransoStruct()
opts.torch_device = device
nvar = getNvarTorch(model.parameters())
opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
opts.opt_tol = 1e-5
# opts.fvalquit = 1e-6
opts.print_level = 1
opts.print_frequency = 1
# opts.maxit = 5
# opts.print_ascii = True

outputs = model(inputs )
acc = (outputs.max(1)[1] == labels).sum().item()/labels.size(0)

print("Initial acc = {}".format(acc)) 

start = time.time()
soln = pygranso(var_spec= model, combined_fn = comb_fn, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))


torch.nn.utils.vector_to_parameters(soln.final.x, model.parameters())
outputs = model(inputs)
acc = (outputs.max(1)[1] == labels).sum().item()/labels.size(0)
print("Train acc = {}".format(acc))