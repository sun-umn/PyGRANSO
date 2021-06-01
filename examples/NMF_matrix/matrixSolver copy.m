function [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec] = matrixSolver(x)
% x is vector
% output all vector: for granso package

X.U = reshape(x(1:4),2,2);
X.V = reshape(x(5:8),2,2);

[f,f_grad] = objectiveFunction_matrix(X);
[ci,ci_grad] = inequalityConstraint_matrix(X);

% give matrix form, output vector form

f_vec = f;

f_grad_vec = [f_grad.U(:);f_grad.V(:)];

ci_vec = [ci(1).U(:)+ci(2).U(:) ; ci(1).V(:) + ci(2).V(:)];

ci_grad_vec = diag([ci_grad(1).U(:);ci_grad(1).V(:)] + [ci_grad(2).U(:);ci_grad(2).V(:)]);

ce_vec = [];

ce_grad_vec = [];

end