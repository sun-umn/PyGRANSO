import time
import numpy as np
import torch
import numpy.linalg as la
from scipy.stats import norm
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso import pygranso
from pygransoStruct import Options, Data
from private.getNvar import getNvarTorch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import pickle

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        
        self.inplanes = 6

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 6, layers[0])
        self.layer2 = self._make_layer(block, 9, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 12, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 15, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(15, num_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None  
   
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)           # 16x16
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # 8x8

        x = self.layer1(x)          # 8x8
        x = self.layer2(x)          # 4x4
        x = self.layer3(x)          # 2x2
        x = self.layer4(x)          # 1x1

        x = self.avgpool(x)         # 1x1
        x = torch.flatten(x, 1)     # remove 1 X 1 grid and make vector of tensor shape 
        x = self.fc(x)

        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def resnet18():
    layers=[2,2,2,2]
    model = ResNet(BasicBlock, layers)
    return model

def mainFun():
    # Please read the documentation on https://pygranso.readthedocs.io/en/latest/

    

    device = torch.device('cuda')
    # device = torch.device('cpu')
    model = resnet18().to(device=device, dtype=torch.double)

    # User defined initialization. 


    # user defined options
    opts = Options()
    nvar = getNvarTorch(model.parameters())
    opts.QPsolver = 'osqp'
    opts.maxit = 300
    opts.opt_tol = 1e-6
    opts.fvalquit = 1e-6
    opts.print_level = 1
    opts.print_frequency = 10
    # opts.halt_on_linesearch_bracket = False
    # opts.max_fallback_level = 3
    # opts.min_fallback_level = 2
    # opts.init_step_size = 1e-2
    # opts.linesearch_maxit = 25
    # opts.is_backtrack_linesearch = True
    # opts.searching_direction_rescaling = True
    # opts.disable_terminationcode_6 = True

    with open('soln_batch10.pkl', 'rb') as f:
        soln_old = pickle.load(f)
    


    ################################################
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 5
    trainset = torchvision.datasets.CIFAR10(
        root='/home/buyun/Documents/GitHub/PyGRANSO/examples/DL_CIFAR10/data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    # data_in
    for i, data in enumerate(trainloader, 0):
        if i >= 1:
            break
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

    print(inputs.shape)
    # variables and corresponding dimensions.
    var_in = {"x_tilde": list(inputs.shape)}

    opts.x0 = torch.reshape(inputs,(torch.numel(inputs),1)) .to(device=device, dtype=torch.double)

    with open('soln_batch10.pkl', 'rb') as f:
        soln_old = pickle.load(f)

    torch.nn.utils.vector_to_parameters(soln_old.final.x, model.parameters())
    outputs = model(inputs.to(device=device, dtype=torch.double))
    acc = (outputs.max(1)[1] == labels.to(
        device=device, dtype=torch.double)).sum().item()/labels.size(0)

    print("Initial acc = {}".format(acc))

 

    data_in = Data()
    data_in.labels = labels.to(device=device)  # label/target [256]
    # input data [256,3,32,32]
    data_in.inputs = inputs.to(device=device, dtype=torch.double)
    data_in.model = model

    # data_in.attack_type = 'L_2'
    data_in.attack_type = 'L_inf'
    # data_in.attack_type = 'perceptual'

    #  main algorithm  
    start = time.time()
    soln = pygranso(var_dim_map = var_in, torch_device = device, user_data = data_in, user_opts = opts)
    end = time.time()
    print("Total Wall Time: {}s".format(end - start))

    final_adv_input = torch.reshape(soln.final.x,inputs.shape)
    adv_outputs2 = model(final_adv_input.to(device=device, dtype=torch.double))
    acc2 = (adv_outputs2.max(1)[1] == labels.to(
        device=device, dtype=torch.double)).sum().item()/labels.size(0)

    print("adv acc final = {}".format(acc2))

    if data_in.attack_type == 'L_2':
        print("adv diff L2 = {}".format( ( torch.norm((inputs.to(device=device, dtype=torch.double) - final_adv_input).reshape(inputs.size()[0], -1)) )))
    elif data_in.attack_type == 'L_inf':
        print("adv diff Linf = {}".format( ( torch.norm((inputs.to(device=device, dtype=torch.double) - final_adv_input).reshape(inputs.size()[0], -1), float('inf') ) )))
    else:
        print("adv diff perceptual = {}".format(torch.norm(final_adv_input-data_in.inputs)))
    

if __name__ == "__main__":
   
    mainFun( )
