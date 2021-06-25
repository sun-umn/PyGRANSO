function [f,f_grad,ci,ci_grad,ce,ce_grad] = combinedFunction(X)
%   combinedFunction: (example_mat/ex1_smoothNMF)
%       Defines objective and inequality constraint functions, with their
%       respective gradients, for an smooth nonnegative matrix factorization optimization problem:
%
%           f(U,V) = || D - UV' ||_F^2,
%           s.t., U >= O, V >= O
%
%       where D is fixed real-valued matrices
%           - D is m by n
%       and U and V are real-valued matrix of the optimization variables
%           - U is m by k
%           - V is n by k
%
%       This example has the following properties:
%           - D = ones(m,n)
%           - m = 4
%           - n = 3
%           - k = 2
%
%       GRANSO's convention is that the inequality constraints must be less
%       than or equal to zero to be satisfied.  For equality constraints,
%       GRANSO's convention is that they must be equal to zero.
%
%   USAGE:
%       [f,f_grad,ci,ci_grad,ce,ce_grad] = ...
%                               combinedFunction(A,B,C,stability_margin,x);
% 
%   INPUT:
%       X                   struct of optimization variables
%                           X.U is a real-valued matrix, m by k
%                           X.V is a real-valued matrix, n by k
%  
%   OUTPUT:
%       f                   value of the objective function at x
%                           scalar real value
% 
%       f_grad              struct of gradient of the objective function at x.
%                           f_grad.U is a real-valued matrix, m by k
%                           f_grad.V is a real-valued matrix, n by k
% 
%       ci                  struct of value of the inequality constraint at x.
%                           c.c1 is a real-valued matrix, m by k
%                           c.c2 is a real-valued matrix, n by k
% 
%       ci_grad             struct of gradient of the inequality constraint at x.
%                           c.c1.U is a real-valued matrix, m*k by m*k
%                           c.c1.V is a real-valued matrix, n*k by m*k
%                           c.c2.U is a real-valued matrix, m*k by n*k
%                           c.c2.V is a real-valued matrix, n*k by n*k
% 
%       ce                  value of the equality constraint(s) at x.
%                           [], since there aren't any equality constraints
%
%       ce_grad             gradient(s) of the equality constraint(s) at x.
%                           [], since there aren't any equality constraints
%
%

% input variable, matirx form
U = X.U;
V = X.V;

D = [1 1 1 1;1 1 1 1;1 1 1 1];

% objective function
f = .5*norm(D - U*V', 'fro')^2;

% gradient of objective function, matrix form
f_grad.U = -(D - U*V') * V;
f_grad.V = -(D - U*V')' * U;

% inequality constraint, matrix form
ci.c1 = -U;
ci.c2 = -V;

% gradient of inequality constraint, matrix form
ci_grad.c1.U = -diag([1 1 1 1 1 1]);
ci_grad.c1.V = zeros(8,6);

ci_grad.c2.U = zeros(6,8);
ci_grad.c2.V = -diag([1 1 1 1 1 1 1 1]);

% equality constraint 
ce = [];
ce_grad = [];

end