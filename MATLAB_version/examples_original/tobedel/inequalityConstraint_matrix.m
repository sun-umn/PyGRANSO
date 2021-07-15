function [ci,ci_grad] = inequalityConstraint_matrix(X)

    x1 = X.x1;
    x2 = X.x2;

    ci.c1 = sqrt(2)*x1-1;
    ci.c2 = 2*x2-1;
    
    
    % # of constr b # of vars of U and V
    ci_grad.c1.x1 = sqrt(2);
    ci_grad.c1.x2 = 0;
    
    ci_grad.c2.x1 = 0;
    ci_grad.c2.x2 = 2;

%     U = X.U;
%     V = X.V;
% 
%     ci.c1 = -U;
%     ci.c2 = -V;
%     
%     
%     % # of constr b # of vars of U and V
%     ci_grad.c1.U = -diag([1 1 1 1 1 1]);
%     ci_grad.c1.V = zeros(4,6);
%     
%     ci_grad.c2.U = zeros(6,4);
%     ci_grad.c2.V = -diag([1 1 1 1]);
    
%     ci = -x;
%     ci_grad = -eye(8);
end