# PyGRANSO

PyGRANSO: a Python numerical package using GRadient-based Algorithm for Non-Smooth Optimization

Toward a user-friendly and scalable numerical package for nonsmooth, nonconvex, constrained optimization

Please check https://pygranso.readthedocs.io/en/latest for detailed documentation.

## Update Log

normalize line search direction. norm(d) = 1, d is the searching direction

If the last fallback strategy failed, make a movement anyway. NOTE: make sure not restore the previous snapshot

Disable the 2nd wolfe condtion so that the line search becomes backtraking line search

Set maxiter = 25 for line search so that there is a lower bound for step size t

reset Hessian every 100 iters helps avoid non-positive definite matrices in BFGS

osqp-GPU is not necessary as it's not the main cost


TODO:

warm start

SR1

Comparison based on
Simple Algorithms for Optimization on Riemannian Manifolds with Constraints

Manopt new autodiff version

## References

Frank E. Curtis, Tim Mitchell, and Michael L. Overton. "A BFGS-SQP method for nonsmooth, nonconvex, constrained optimization and its evaluation using relative minimization profiles." Optimization Methods and Software 32.1 (2017): 148-181.

## Contact
Codes written by Buyun Liang (https://www.linkedin.com/in/buyun-liang/). Questions or bug reports please send email to Buyun Liang, liang664@umn.edu.

Thanks to bug reporters: 
