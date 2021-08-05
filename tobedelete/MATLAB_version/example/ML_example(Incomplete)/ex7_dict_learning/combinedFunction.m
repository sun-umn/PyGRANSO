function [f,f_grad,ci,ci_grad,ce,ce_grad] = combinedFunction(X,parameters)
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

n_samples = parameters.n_samples;
n_components = parameters.n_components;
n_features = parameters.n_features;
alpha = parameters.alpha;

Y = parameters.Y;

% objective function
f = .5*norm(Y - U*V, 'fro')^2 + alpha * norm(U, 1);

% first part of gradient of objective function, matrix form
f_grad.U = -(Y - U*V) * V';
f_grad.V = -(Y - U*V)' * U;

% (sub)gradient of L1 norm
g = ones(n_features, n_components);
g(U < 0) = -1;

indicator = zeros(1,n_components);

for j = 1:n_components
    if (norm(U, 1) == norm(U(:,j), 1))
        indicator(j) = 1;
        break;
    end
end

f_grad.U = f_grad.U + indicator .* g;

% % inequality constraint, matrix form
% ci.c1 = -U;
% ci.c2 = -V;
% 
ci = [];

% % gradient of inequality constraint, matrix form
% ci_grad.c1.U = -speye(n_features * n_components);
% ci_grad.c1.V = zeros(n_components * n_samples,n_features * n_components);
% 
% ci_grad.c2.U = zeros(n_features * n_components,n_components * n_samples);
% ci_grad.c2.V = -speye(n_components * n_samples);
ci_grad = [];

% equality constraint 
ce = [];
% for k = 1:n_components
%     ce.(strcat('c',int2str(k)) ) = norm(V(k,:),2)-1;
% end
ce_grad = [];

end