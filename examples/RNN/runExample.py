import time
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
## Adding training data directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO/examples/DL_CIFAR10')
sys.path.append('/home/buyun/Documents/GitHub/expRNN')

from pygranso import pygranso
from pygransoStruct import Options, Data
from private.getNvar import getNvarTorch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets
# from orthogonal import OrthogonalRNN
# from initialization import cayley_init_
# from trivializations import expm
import pickle
from torchvision.transforms import ToTensor



sequence_length = 28
input_size = 28
hidden_size = 30
num_layers = 1
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# setting device on GPU if available, else CPU
device = torch.device('cuda')
# device = torch.device('cpu')

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
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device, dtype=torch.double)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device, dtype=torch.double)
        # Passing in the input and hidden state into the model and  obtaining outputs
        # out, hidden = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, hidden = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)


        #Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out[:, -1, :])
        return out
       

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def mainFun():


        torch.manual_seed(0)
        
        print('Using device:', device)
        model = RNN(input_size, hidden_size, num_layers, num_classes).to(device=device, dtype=torch.double)
        model.train()
        ################### PyGRANSO
        train_data = datasets.MNIST(
            root = '/home/buyun/Documents/GitHub/PyGRANSO/mnist',
            train = True,                         
            transform = ToTensor(), 
            download = False,            
        )
        test_data = datasets.MNIST(
            root = '/home/buyun/Documents/GitHub/PyGRANSO/mnist', 
            train = False, 
            transform = ToTensor()
        )

        loaders = {
            'train' : torch.utils.data.DataLoader(train_data, 
                                                batch_size=100, 
                                                shuffle=True, 
                                                num_workers=1),
            
            'test'  : torch.utils.data.DataLoader(test_data, 
                                                batch_size=100, 
                                                shuffle=True, 
                                                num_workers=1),
        }

        inputs, labels = next(iter(loaders['train']))
        inputs, labels = inputs.reshape(-1, sequence_length, input_size).to(device=device, dtype=torch.double), labels.to(device=device)

        var_in = {}
        var_count = 0
        var_str = "x"
        for i in model.parameters():
            # print(i.shape)
            var_in[var_str+str(var_count)]= list(i.shape)
            var_count += 1

        # variables and corresponding dimensions.
        

        # data_in
        data_in = Data()
        data_in.labels = labels
        data_in.inputs = inputs
        data_in.model = model
        data_in.hidden_size = hidden_size

        

        opts = Options()
        nvar = getNvarTorch(model.parameters())
        opts.QPsolver = 'osqp' 
        opts.maxit = 100
        opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
        opts.opt_tol = 1e-6
        opts.fvalquit = 1e-6
        opts.print_level = 1
        opts.print_frequency = 10
        # opts.print_ascii = True



        # opts.max_fallback_level = 3
        # opts.min_fallback_level = 2
        # opts.init_step_size = 1e-2
        opts.init_step_size = 1e-1
        opts.halt_on_linesearch_bracket = False
        # opts.disable_terminationcode_6 = True

        opts.linesearch_maxit = 25
        # opts.linesearch_maxit = 10
        opts.is_backtrack_linesearch = True
        opts.searching_direction_rescaling = True
        opts.limited_mem_size = 200

            

        # with open('orthogonalRNN_300iter.pkl', 'rb') as f:
        #     soln_old = pickle.load(f)
        # opts.x0 = soln_old.final.x
        # torch.nn.utils.vector_to_parameters(soln_old.final.x, model.parameters())

        logits = model(inputs)
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        print("Initial acc = {:.2f}%".format((100 * correct/len(inputs))))  

        #  main algorithm  
        start = time.time()
        soln = pygranso(var_dim_map = var_in, torch_device = device, user_data = data_in, user_opts = opts)
        end = time.time()

        # numpyVec2DLTorchTensor(soln.final.x,model) # update model paramters
        torch.nn.utils.vector_to_parameters(soln.final.x, model.parameters())
        logits = model(inputs)
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        print("Final acc = {:.2f}%".format((100 * correct/len(inputs))))     
        
        print("total time = {} s".format(end-start))

        save_object(soln, 'orthogonalRNN_iter500_hidden30.pkl')



if __name__ == "__main__":
    mainFun()