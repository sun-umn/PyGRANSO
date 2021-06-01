function [ci,ci_grad] = inequalityConstraint_matrix(X)
 
    U = X.U;
    V = X.V;

    ci.c1 = -U;
    ci.c2 = -V;
    
    
    % # of constr b # of vars of U and V
    ci_grad.c1.U = -diag([1 1 1 1]);
    ci_grad.c1.V = diag([0 0 0 0]);
    
    ci_grad.c2.U = diag([0 0 0 0]);
    ci_grad.c2.V = -diag([1 1 1 1]);
    
%     ci = -x;
%     ci_grad = -eye(8);
end