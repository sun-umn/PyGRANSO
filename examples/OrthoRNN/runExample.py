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
from orthogonal import OrthogonalRNN
from initialization import cayley_init_
from trivializations import expm
import pickle




class Model(nn.Module):
    def __init__(self, hidden_size, permute):
        super(Model, self).__init__()
        self.permute = permute
        permute = np.random.RandomState(92916)
        self.register_buffer("permutation", torch.LongTensor(permute.permutation(784)))
        self.rnn = OrthogonalRNN(1, hidden_size, initializer_skew=cayley_init_, mode=("dynamic", 100, 100), param=expm)

        self.lin = nn.Linear(hidden_size, 10)
        self.loss_func = nn.CrossEntropyLoss()


    def forward(self, inputs):
        if self.permute:
            inputs = inputs[:, self.permutation]

        if isinstance(self.rnn, OrthogonalRNN):
            state = self.rnn.default_hidden(inputs[:, 0, ...])
        else:
            state = (torch.zeros((inputs.size(0), self.hidden_size), device=inputs.device),
                     torch.zeros((inputs.size(0), self.hidden_size), device=inputs.device))
        for input in torch.unbind(inputs, dim=1):
            out_rnn, state = self.rnn(input.unsqueeze(dim=1), state)
            if isinstance(self.rnn, nn.LSTMCell):
                state = (out_rnn, state)
        return self.lin(state)

    def loss(self, logits, y):
        return self.loss_func(logits, y)

    def correct(self, logits, y):
        return torch.eq(torch.argmax(logits, dim=1), y).float().sum()

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def mainFun():


        torch.manual_seed(0)
        # setting device on GPU if available, else CPU
        device = torch.device('cuda')
        # device = torch.device('cpu')
        print('Using device:', device)
        model = Model(170, False).to(device=device, dtype=torch.double)
        model.train()
        ################### PyGRANSO
        kwargs = {'num_workers': 1, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=128, shuffle=True, **kwargs)

        inputs, labels = next(iter(train_loader))
        inputs, labels = inputs.to(device=device, dtype=torch.double).view(-1, 784), labels.to(device=device)

        var_in = {}
        var_count = 0
        var_str = "x"
        for i in model.parameters():
            # print(i.data)
            var_in[var_str+str(var_count)]= list(i.shape)
            var_count += 1

        # variables and corresponding dimensions.
        

        # data_in
        data_in = Data()
        data_in.labels = labels
        data_in.inputs = inputs
        data_in.model = model

        

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

            

        # with open('orthogonalRNN_300iter.pkl', 'rb') as f:
        #     soln_old = pickle.load(f)
        # opts.x0 = soln_old.final.x
        # torch.nn.utils.vector_to_parameters(soln_old.final.x, model.parameters())

        logits = model(inputs)
        correct = model.correct(logits, labels)
        print("Initial acc = {:.2f}%".format((100 * correct/len(inputs)).item()))  

        #  main algorithm  
        start = time.time()
        soln = pygranso(var_dim_map = var_in, torch_device = device, user_data = data_in, user_opts = opts)
        end = time.time()

        # numpyVec2DLTorchTensor(soln.final.x,model) # update model paramters
        torch.nn.utils.vector_to_parameters(soln.final.x, model.parameters())
        logits = model(inputs)
        correct = model.correct(logits, labels)
        print("Final acc = {:.2f}%".format((100 * correct/len(inputs)).item()))     
        
        print("total time = {} s".format(end-start))

        save_object(soln, 'orthogonalRNN.pkl')


if __name__ == "__main__":
    mainFun()