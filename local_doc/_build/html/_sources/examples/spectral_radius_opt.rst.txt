Spectral Radius Optimization
========

This example is from Curtis, Frank E., Tim Mitchell, and Michael L. Overton. "A BFGS-SQP method for nonsmooth, nonconvex, constrained optimization and its evaluation using relative minimization profiles." Optimization Methods and Software 32.1 (2017): 148-181.


.. image:: images/specR.png
   :width: 600


runExample.py
-----------------

First read the prepared input data from a Matlab file::

      # device = torch.device('cuda' )
      device = torch.device('cpu' )
      print('Using device:', device)

      # read input data from matlab file
      currentdir = os.path.dirname(os.path.realpath(__file__))
      file = currentdir + "/spec_radius_opt_data.mat"
      mat = scipy.io.loadmat(file)
      mat_struct = mat['sys']
      mat_struct = mat_struct[0,0]
      A = torch.from_numpy(mat_struct['A']).to(device=device, dtype=torch.double)
      B = torch.from_numpy(mat_struct['B']).to(device=device, dtype=torch.double)
      C = torch.from_numpy(mat_struct['C']).to(device=device, dtype=torch.double)
      p = B.shape[1]
      m = C.shape[0]

The arguments for ``pygranso()`` is ``var_dim_map`` (if specify it, please leave nn_model as default None), ``nn_model`` (only used in deep learning problem. If specify it, please leave var_dim_map as default None), ``torch_device`` (optional, default torch.device('cpu')), ``user_data`` (optional) and ``user_opts`` (optional).

1. ``var_dim_map``

   ``var_in`` is a python dictionary used for indicate variable name and corresponding matrix dimension. 
   Since ``X`` is a matrix here, we set the dimension to ``(p,m)``::

      var_in = {"X": (p,m) }

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

      data_in = Data()
      data_in.A = A
      data_in.B = B
      data_in.C = C
      data_in.stability_margin = 1

4. ``user_opts``

   User-provided options. First initialize a structure for options::

      from pygransoStruct import Options
      opts = Options()

   Then define the options::

      opts.QPsolver = 'osqp' 
      opts.maxit = 200
      opts.x0 = torch.zeros(p*m,1).to(device=device, dtype=torch.double)
      opts.print_level = 1
      opts.print_frequency = 1

   See :ref:`settings<settings>` for more information.

Call the main function::

   soln = pygranso(var_dim_map = var_in, torch_device = device, user_data = data_in, user_opts = opts)

combinedFunction.py
-----------------

In ``combinedFunction.py`` , ``combinedFunction(X_struct, data_in = None)`` is used to generate user defined objection function ``f``, 
inequality constraint function ``ci`` and equality constraint function ``ce``.

Notice that we have auto-differentiation feature implemented, so the analytical gradients are not needed.

1. Obtain the (pytorch) tensor form variables from structure ``X_struct``. And require gradient for the autodiff::

      X = X_struct.X
      X.requires_grad_(True)

2. Obtain data from ``runExample.py``::

      A = data_in.A
      B = data_in.B
      C = data_in.C
      stability_margin = data_in.stability_margin

3. Define objective function. Notice that we must use pytorch function::

      M           = A + B@X@C
      [D,_]       = LA.eig(M)
      f = torch.max(D.imag)

4. Define the inequality constraint function. We must initialize ``ci`` as a struct, 
   then assign different constraints as ``ci.c1``, ``ci.c2``, ``ci.c3``...::

      ci = general_struct()
      ci.c1 = torch.max(D.real) + stability_margin

5. Since no inequality constraint required in this problem, we set ``ce`` to ``None``::

      ce = None

6. Return user-defined results::

     return [f,ci,ce]

``eval_obj(X_struct,data_in = None)`` is similar to ``combinedFunction()`` described above. The only difference is that this function is only used to generate objective value. 
