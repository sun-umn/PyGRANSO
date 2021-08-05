function [f,f_grad,ci,ci_grad,ce,ce_grad] = combinedFunction(X)
%   combinedFunction: (examples/ex1)
%       Defines objective and inequality constraint functions, with their
%       respective gradients.
%       
%       GRANSO will minimize the objective function.  The gradient must be
%       a column vector.
% 
%   USAGE:
%       [f,f_grad,ci,ci_grad,ce,ce_grad] = combinedFunction(X);
%
%   INPUT:
%       X                   struct of optimization variables
%                           X.u is a scalar (real-valued matrix, 1 by 1)
%                           X.v is a scalar (real-valued matrix, 1 by 1)
%  
%   OUTPUT:
%       f                   value of the objective function at X
%                           scalar real value
% 
%       f_grad              struct of gradient of the objective function at x.
%                           f_grad.u is a scalar (real-valued matrix, 1 by 1)
%                           f_grad.v is a scalar (real-valued matrix, 1 by 1)
% 
%       ci                  struct of value of the inequality constraint at x.
%                           1st constraint: c.c1 is a scalar (real-valued matrix, 1 by 1)
%                           2nd constraint: c.c2 is a scalar (real-valued matrix, 1 by 1)
% 
%       ci_grad             struct of gradient of the inequality constraint at x.
%                           c.c1.u is a scalar (real-valued matrix, 1 by 1)
%                           c.c1.v is a scalar (real-valued matrix, 1 by 1)
%                           c.c2.u is a scalar (real-valued matrix, 1 by 1)
%                           c.c2.v is a scalar (real-valued matrix, 1 by 1)
% 
%       ce                  value of the equality constraint(s) at x.
%                           [], since there aren't any equality constraints
%
%       ce_grad             gradient(s) of the equality constraint(s) at x.
%                           [], since there aren't any equality constraints
%

    u = X.u;
    v = X.v; 

    f = abs(u) + v^2;

    % GRADIENT AT X
    % Compute the 2nd term
    f_grad.v = 2*v;
    % Add in the 1st term, where we must handle the sign due to the 
    % absolute value
    if u >= 0
      f_grad.u    = 1;
    else
      f_grad.u    = -1;
    end
    
    
    ci = [];
    
    % # of constr b # of vars of U and V
    ci_grad = [];

    ce = [];
    ce_grad = [];
          
end