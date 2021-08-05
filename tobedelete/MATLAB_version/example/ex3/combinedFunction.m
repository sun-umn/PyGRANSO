function [f,f_grad,ci,ci_grad,ce,ce_grad] = combinedFunction(X,parameters)
%   combinedFunction: (examples/ex3)
%       Defines objective and inequality constraint functions, with their
%       respective gradients.
%       
%       GRANSO will minimize the objective function.  The gradient must be
%       a column vector.
% 
%   USAGE:
%       [f,f_grad,ci,ci_grad,ce,ce_grad] = combinedFunction(w,X);
%
%   INPUT:
%       parameters           
%                           strcut of constant for nonsmooth Rosenbrock function 
%                           parameters.w is a real finite scalar value
%
%       X                   struct of optimization variables
%                           X.x1 is a scalar (real-valued matrix, 1 by 1)
%                           X.x2 is a scalar (real-valued matrix, 1 by 1)
%  
%   OUTPUT:
%       f                   value of the objective function at X
%                           scalar real value
% 
%       f_grad              struct of gradient of the objective function at x.
%                           f_grad.x1 is a scalar (real-valued matrix, 1 by 1)
%                           f_grad.x2 is a scalar (real-valued matrix, 1 by 1)
% 
%       ci                  struct of value of the inequality constraint at x.
%                           1st constraint: c.c1 is a scalar (real-valued matrix, 1 by 1)
%                           2nd constraint: c.c2 is a scalar (real-valued matrix, 1 by 1)
% 
%       ci_grad             struct of gradient of the inequality constraint at x.
%                           c.c1.x1 is a scalar (real-valued matrix, 1 by 1)
%                           c.c1.x2 is a scalar (real-valued matrix, 1 by 1)
%                           c.c2.x1 is a scalar (real-valued matrix, 1 by 1)
%                           c.c2.x2 is a scalar (real-valued matrix, 1 by 1)
% 
%       ce                  value of the equality constraint(s) at x.
%                           [], since there aren't any equality constraints
%
%       ce_grad             gradient(s) of the equality constraint(s) at x.
%                           [], since there aren't any equality constraints
%

    w = parameters.w;
    
    x1 = X.x1;
    x2 = X.x2;
 

    f = w*abs(x1^2 - x2) + (1 - x1)^2;

    % GRADIENT AT X
    % Compute the 2nd term
    f_grad.x1      = -2*(1-x1);
    f_grad.x2 = 0;
    % Add in the 1st term, where we must handle the sign due to the 
    % absolute value
    if x1^2 - x2 >= 0
      f_grad.x1    = f_grad.x1 + w*2*x1;
      f_grad.x2    = f_grad.x2 - w;
    else
      f_grad.x1    = f_grad.x1 - w*2*x1;
      f_grad.x2    = f_grad.x2 + w;
    end
    
    
    ci.c1 = sqrt(2)*x1-1;
    ci.c2 = 2*x2-1;
    
    
    % # of constr b # of vars of U and V
    ci_grad.c1.x1 = sqrt(2);
    ci_grad.c1.x2 = 0;
    
    ci_grad.c2.x1 = 0;
    ci_grad.c2.x2 = 2;

    ce = [];
    ce_grad = [];
          
end