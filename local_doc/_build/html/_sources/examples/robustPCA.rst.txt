Robust PCA
========

This example is based on Reference: Yi, Xinyang, et al. "Fast algorithms for robust PCA via gradient descent." Advances in neural information processing systems. 2016.


.. image:: images/robustPCA.png
   :width: 600


runExample.py
-----------------

The arguments for ``pygranso()`` is ``var_dim_map`` (if specify it, please leave nn_model as default None), ``nn_model`` (only used in deep learning problem. If specify it, please leave var_dim_map as default None), ``torch_device`` (optional, default torch.device('cpu')), ``user_data`` (optional) and ``user_opts`` (optional).

1. ``var_dim_map``
   
  ``var_in`` is a python dictionary used for indicate variable name and corresponding matrix dimension. 
   Since ``M`` and ``S`` are two matrices here, we set both of their dimension to ``(d1,d2)``::

      d1 = 7
      d2 = 8
      var_in = {"M": (d1,d2),"S": (d1,d2)}

2. ``torch_device``
   
   In the example, we will use cpu. (recommend cpu for small scale problem)::

      device = torch.device('cpu')

3. ``user_data``

   To save the computational sources, we recommend to generate all the required paramters in the ``runExample.py``.

   .. warning::
      All non-scalar parameters should be Pytorch tensor
   
   First initialize a structure for Data::

      from pygransoStruct import Data
      data_in = Data()

   Then define the parameters::

      torch.manual_seed(1)
      parameters.eta = .5
      data_in.Y = torch.randn(d1,d2).to(device=device, dtype=torch.double)

4. ``user_opts``

   User-provided options. First initialize a structure for options::

      from pygransoStruct import Options
      opts = Options()

   Then define the options::
      
      opts.x0 = .2 * torch.ones((2*d1*d2,1)).to(device=device, dtype=torch.double)

   See :ref:`settings<settings>` for more information.

Call the main function::

   soln = pygranso(var_dim_map = var_in, torch_device = device, user_data = data_in, user_opts = opts)

combinedFunction.py
-----------------

In ``combinedFunction.py`` , ``combinedFunction(X_struct, data_in = None)`` is used to generate user defined objection function ``f``, 
inequality constraint function ``ci`` and equality constraint function ``ce``.

Notice that we have auto-differentiation feature implemented, so the analytical gradients are not needed.

1. Obtain the (pytorch) tensor form variables from structure ``X_struct``. And require gradient for the autodiff::

      M = X_struct.M
      S = X_struct.S
      M.requires_grad_(True)
      S.requires_grad_(True)

2. Obtain data from ``runExample.py``::

       eta = data_in.eta
       Y = data_in.Y

3. Define objective function. Notice that we must use pytorch function::

      f = torch.norm(M, p = 'nuc') + eta * torch.norm(S, p = 1)

4. Since no inequality constraint required in this problem, we set ``ci`` to ``None``::

      ci = None   

5. Define the equality constraint function. We must initialize ``ce`` as a struct, 
   then assign different constraints as ``ce.c1``, ``ce.c2``, ``ce.c3``...::

      from pygransoStruct import general_struct
      ce = general_struct()
      ce.c1 = M + S - Y

6. Return user-defined results::

     return [f,ci,ce]

``eval_obj(X_struct,data_in = None)`` is similar to ``combinedFunction()`` described above. The only difference is that this function is only used to generate objective value. 
