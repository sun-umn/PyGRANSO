import time
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append(r'C:\Users\Buyun\Documents\GitHub\PyGRANSO')
## Adding training data directories. Should be modified by user
sys.path.append(r'C:\Users\Buyun\Documents\GitHub\PyGRANSO\examples\DL_CIFAR10')
from pygranso import pygranso
from pygransoStruct import Options, Data
from private.getNvar import getNvarTorch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

def mainFun():

        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        batch_size = 1000
        trainset = torchvision.datasets.CIFAR10(root='C:/Users/Buyun/Documents/GitHub/PyGRANSO/examples/DL_CIFAR10/data', train=True, download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

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

        torch.manual_seed(0)
        # setting device on GPU if available, else CPU
        device = torch.device('cuda')
        # device = torch.device('cpu')
        print('Using device:', device)
        model = Net().to(device=device, dtype=torch.double)
        ################### PyGRANSO

        # data_in
        for i, data in enumerate(trainloader, 0):        
                if i >= 1:
                        break   
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

        data_in = Data()
        data_in.labels = labels.to(device=device ) # label/target [256]
        data_in.inputs = inputs.to(device=device, dtype=torch.double) # input data [256,3,32,32]

        opts = Options()
        nvar = getNvarTorch(model.parameters())
        opts.QPsolver = 'osqp' 
        opts.maxit = 100
        opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
        opts.opt_tol = 1e-6
        opts.fvalquit = 1e-6
        opts.print_level = 1
        opts.print_frequency = 1
        # opts.print_ascii = True
        # opts.wolfe1 = 0.1
        # opts.wolfe2 = 1e-4
        opts.halt_on_linesearch_bracket = False
        opts.max_fallback_level = 3
        opts.min_fallback_level = 2
        opts.init_step_size = 1e-2
        opts.linesearch_maxit = 25
        opts.is_backtrack_linesearch = True
        opts.searching_direction_rescaling = True
        opts.disable_terminationcode_6 = True

        outputs = model(inputs.to(device=device, dtype=torch.double) )
        acc = (outputs.max(1)[1] == labels.to(device=device, dtype=torch.double) ).sum().item()/labels.size(0)

        print("Initial acc = {}".format(acc))        

        #  main algorithm  
        start = time.time()
        soln = pygranso(user_data = data_in, user_opts = opts, nn_model = model, torch_device = device)
        end = time.time()

        # numpyVec2DLTorchTensor(soln.final.x,model) # update model paramters
        torch.nn.utils.vector_to_parameters(soln.final.x, model.parameters())
        outputs = model(inputs.to(device=device, dtype=torch.double) )
        acc = (outputs.max(1)[1] == labels.to(device=device, dtype=torch.double) ).sum().item()/labels.size(0)

        print("acc = {}".format(acc))
        print("total time = {} s".format(end-start))


if __name__ == "__main__":
    mainFun()