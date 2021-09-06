# PyGRANSO

PyGRANSO: a Python numerical package using GRadient-based Algorithm for Non-Smooth Optimization

Toward a user-friendly and scalable numerical package for nonsmooth, nonconvex, constrained optimization

Please check https://pygranso.readthedocs.io/en/latest for detailed documentation.

## Update Log

normalize line search direction. norm(d) = 1, d is the searching direction

If the last fallback strategy failed, make a movement anyway. NOTE: make sure not restore the previous snapshot

Diablae the 2nd wolfe condtion so that the line search becomes backtraking line search

Set maxiter = 25 for line search so that there is a lower bound for step size t

TODO:

profiling

osqp-GPU


## References

Frank E. Curtis, Tim Mitchell, and Michael L. Overton. "A BFGS-SQP method for nonsmooth, nonconvex, constrained optimization and its evaluation using relative minimization profiles." Optimization Methods and Software 32.1 (2017): 148-181.

## Contact
Codes written by Buyun Liang (https://www.linkedin.com/in/buyun-liang/). Questions or bug reports please send email to Buyun Liang, liang664@umn.edu.

Thanks to bug reporters: 
