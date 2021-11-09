New Parameters
========================

QPsolver
------------------

String in {'osqp', 'gurobi'}. Default value: 'osqp'

Select the QP solver used in the steering strategy and termination condition.

init_step_size     
----------------        

Positive real value. Default value: 1

Initial step size t in line search method. Recommend using small value (e.g., 1e-2) for deep learning problem.

init_step_size     
----------------        

Positive integer. Default value: inf

Max number of iterations in line search method. Recommend using small value (e.g., 25) for deep learning problem.

is_backtrack_linesearch     
----------------          

Boolean value. Default value: False

By default, PyGRANSO will use Weak-Wolfe line search method. By enabling this method, the second wolfe condition will be disabled.

searching_direction_rescaling     
----------------          

Boolean value. Default value: False

Rescale the norm of searching direction to be 1. Recommend setting True in deep learning problem.

disable_terminationcode_6     
----------------          

Boolean value. Default value: False

Disable termination code 6 to ensure pygranso can always make a movement even if the line search failed. Recommend setting True in deep learning problem.



