function [f,f_grad,ci,ci_grad,ce,ce_grad] = combinedFunction(X, parameters)
%   combinedFunction: (example_mat/ex2_spectral_radius_optimization)
%       Defines objective and inequality constraint functions, with their
%       respective gradients, for an eigenvalue optimization problem of a 
%       static-output-feedback (SOF) plant:
%       
%           M = A + BXC,
%
%       where A,B,C are all fixed real-valued matrices
%           - A is n by n 
%           - B is n by p
%           - C is m by n
%       and X is a real-valued p by m matrix of the optimization variables.
%
%   USAGE:
%       [f,f_grad,ci,ci_grad,ce,ce_grad] = ...
%                               combinedFunction(X);
%
%   INPUT:
%       X                   struct of optimization variables
%                           X.XX is a real-valued matrix, p by m
%
%   OUTPUT:
%       f                   value of the objective function at x
%                           scalar real value
%
%       f_grad              struct of gradient of the objective function at x.
%                           f_grad.XX is a real-valued matrix, p by m 
%
%       ci                  struct of value of the inequality constraint at x.
%                           c.c1 is the value of the inequality constraint at x.
%                           scalar real value
%
%       ci_grad             struct of gradient of the inequality constraint at x.
%                           c.c1.XX is a real-valued matrix,  p*m by 1 
%
%       ce                  value of the equality constraint(s) at x.
%                           [], since there aren't any equality constraints
%
%       ce_grad             gradient(s) of the equality constraint(s) at x.
%                           [], since there aren't any equality constraints
%
%

XX           = X.XX;

% [A,B,C,~,~] = loadExample('ex4_data_n=200.mat');
A = parameters.A;
B = parameters.B;
C = parameters.C;
stability_margin = 1;
M           = A + B*XX*C;
[V,D]       = eig(M);
d           = diag(D);
[VL,D_conj] = eig(M');
dl          = conj(diag(D_conj));

% OBJECTIVE VALUE AT X
% Get the max imaginary part, and an eigenvalue associated with it,
% since the constraint is to limit eigenvalues to a band centered on
% the x-axis
[mi,indx]   = max(imag(d));
indx        = indx(1);
lambda      = d(indx);
f           = mi;

% GRADIENT OF THE OBJECTIVE AT X
% Get its corresponding right eigenvector
x           = V(:,indx);
% Now find the matching left eigenvector for lambda
[~,indx]    = min(abs(dl - lambda));
y           = VL(:,indx);
Bty         = B.'*y;
Cx          = C*x;
% Gradient of inner product with respect to A
% f_grad.XX is a real-valued matrix, p by m
f_grad.XX      = imag((conj(Bty)*Cx.')/(y'*x));



% INEQUALITY CONSTRAINT AT X
% Compute the spectral abscissa of A from the spectrum and an
% eigenvalue associated with it
[ci.c1,indx]   = max(real(d));
indx        = indx(1);
lambda      = d(indx);
% account for the stability margin in the inequality constraint
ci.c1          = ci.c1 + stability_margin;

% GRADIENT OF THE INEQUALITY CONSTRAINT AT X
% Get its corresponding right eigenvector
x           = V(:,indx);
% Now find the matching left eigenvector for lambda
[~,indx]    = min(abs(dl - lambda));
indx        = indx(1);
y           = VL(:,indx);
Bty         = B.'*y;
Cx          = C*x;
% Gradient of inner product with respect to A
ci_grad.c1.XX     = real((conj(Bty)*Cx.')/(y'*x));
ci_grad.c1.XX = ci_grad.c1.XX(:);

% EQUALITY CONSTRAINT
% Return [] when (in)equality constraints are not present.
ce = [];
ce_grad=[];
end

function [A,B,C,p,m] = loadExample(filename)
% MATLAB does not allow loading variable names into a function scope
% that has nested functions so we need a completely separate function
% for this task.
load(filename);
A   = sys.A;
B   = sys.B;
C   = sys.C;
p   = size(B,2);
m   = size(C,1);
end