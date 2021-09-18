Generalized LASSO
========

Generalized LASSO: total variation denoising

Reference: Boyd, Stephen, Neal Parikh, and Eric Chu. Distributed optimization and statistical learning via the alternating direction method of multipliers. Now Publishers Inc, 2011.

.. image:: images/tvDenoising.png
   :width: 600


runExample.py
-----------------

The arguments for ``pygranso()`` is ``var_dim_map`` (if specify it, please leave nn_model as default None), ``nn_model`` (only used in deep learning problem. If specify it, please leave var_dim_map as default None), ``torch_device`` (optional, default torch.device('cpu')), ``user_data`` (optional) and ``user_opts`` (optional).

1. ``var_dim_map``
   
   ``var_in`` is a python dictionary used for indicate variable name and corresponding matrix dimension. 
   Since ``x`` is a vector here, we set the dimension to ``(n,1)``::

      n = 80
      var_in = {"x": (n,1)}

2. ``torch_device``
   
   In the example, we will use cpu. (recommend cpu for small scale problem)::

      device = torch.device('cpu')
      
3. ``user_data``

   To save the computational sources, we recommend to generate all the required paramters in the ``runExample.py`` and 
   pass it to ``combinedFunction.py.`` through function ``pygranso()``.

   .. warning::
      All non-scalar parameters should be Pytorch tensor
   
   First initialize a structure for data::

      from pygransoStruct import Data
      data_in = Data()

   Then define the parameters::

      eta = 0.5 # parameter for penalty term
      torch.manual_seed(1)
      b = torch.rand(n,1)
      pos_one = torch.ones(n-1)
      neg_one = -torch.ones(n-1)
      F = torch.zeros(n-1,n)
      F[:,0:n-1] += torch.diag(neg_one,0) 
      F[:,1:n] += torch.diag(pos_one,0)
      data_in.F = F.to(device=device, dtype=torch.double)  # double precision requireed in torch operations 
      data_in.b = b.to(device=device, dtype=torch.double)
      data_in.eta = np.double(eta) # double precision requireed in torch operations 

4. ``user_opts``

   User-provided options. First initialize a structure for options::

      from pygransoStruct import Options
      opts = Options()

   Then define the options::

      opts.QPsolver = 'osqp' 
      opts.maxit = 1000
      opts.x0 = torch.ones((n,1)).to(device=device, dtype=torch.double)
      opts.print_level = 1
      opts.print_frequency = 10

   See :ref:`settings<settings>` for more information.

Call the main function::

   soln = pygranso(var_dim_map = var_in, torch_device = device, user_data = data_in, user_opts = opts)

combinedFunction.py
-----------------

In ``combinedFunction.py`` , ``combinedFunction(X_struct, data_in = None)`` is used to generate user defined objection function ``f``, 
inequality constraint function ``ci`` and equality constraint function ``ce``.

Notice that we have auto-differentiation feature implemented, so the analytical gradients are not needed.

1. Obtain the (pytorch) tensor form variables from structure ``X_struct``. And require gradient for the autodiff::

      x = X_struct.x
      x.requires_grad_(True)

2. Obtain data from ``runExample.py``::

       b = data_in.b
       F = data_in.F
       eta = data_in.eta

3. Define objective function. Notice that we must use pytorch function::

      f = (x-b).t() @ (x-b)  + eta * torch.norm( F@x, p = 1)

4. Since no equality constraint required in this problem, we set ``ci`` to ``None``::

      ci = None   

5. Since no inequality constraint required in this problem, we set ``ci`` to ``None``::

      ce = None

6. Return user-defined results::

     return [f,ci,ce]

``eval_obj(X_struct,data_in = None)`` is similar to ``combinedFunction()`` described above. The only difference is that this function is only used to generate objective value. 
