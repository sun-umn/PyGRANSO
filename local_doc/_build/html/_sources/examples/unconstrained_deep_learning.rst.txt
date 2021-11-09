Unconstrained Deep Learning
========

Generalized LASSO: total variation denoising

Reference: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

.. image:: images/lenet5.png
   :width: 600


runExample.py
-----------------

The arguments for ``pygranso()`` is ``var_dim_map`` (if specify it, please leave nn_model as default None), ``nn_model`` (only used in deep learning problem. If specify it, please leave var_dim_map as default None), ``torch_device`` (optional, default torch.device('cpu')), ``user_data`` (optional) and ``user_opts`` (optional).

1. ``nn_model``

   Check Reference for the initialization of neural network in PyTorch::
   
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

2. ``torch_device``
   
   In the example, we will use cuda. (If cuda is not available, please use cpu instead)::

      device = torch.device('cuda')
   

3. ``user_data``

   Prior to assigning ``data_in``, let's load the data::
         
         transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
         batch_size = 1000
         trainset = torchvision.datasets.CIFAR10(root='C:/Users/Buyun/Documents/GitHub/PyGRANSO/examples/DL_CIFAR10/data', train=True, download=False, transform=transform)
         trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
         # data_in
         for i, data in enumerate(trainloader, 0):        
             if i >= 1:
                     break   
             # get the inputs; data is a list of [inputs, labels]
             inputs, labels = data

   To save the computational sources, we recommend to generate all the required data in the ``runExample.py``.

   .. warning::
      All non-scalar parameters should be in Pytorch tensor form
   
   First initialize a structure for data::

      from pygransoStruct import Data
      data_in = Data()

   Then define the data::

      data_in.labels = labels.to(device=device ) # label/target [256]
      data_in.inputs = inputs.to(device=device, dtype=torch.double) # input data [256,3,32,32]

4. ``user_opts``

   User-provided options. First initialize a structure for options::

      from pygransoStruct import Options
      opts = Options()

   Then define the options::

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

   See :ref:`settings<settings>` for more information.

Call the main function::

     soln = pygranso(nn_model = model, torch_device = device, user_data = data_in, user_opts = opts)

combinedFunction.py
-----------------

In ``combinedFunction.py`` , ``combinedFunction(X_struct, data_in = None)`` is used to generate user defined objection function ``f``, 
inequality constraint function ``ci`` and equality constraint function ``ce``.

Notice that we have auto-differentiation feature implemented, so the analytical gradients are not needed.

1. Obtain data from ``runExample.py``::
   
       inputs = data_in.inputs
       labels = data_in.labels

2. Define objective function. Notice that we must use pytorch function::

      outputs = model(inputs)
      criterion = nn.CrossEntropyLoss()
      f = criterion(outputs, labels)

3. Since no inequality constraint required in this problem, we set ``ci`` to ``None``::

      ci = None   

4. Since no equality constraint required in this problem, we set ``ce`` to ``None``::

      ce = None 

5. Return user-defined results::

     return [f,ci,ce]

``eval_obj(X_struct,data_in = None)`` is similar to ``combinedFunction()`` described above. The only difference is that this function is only used to generate objective value. 