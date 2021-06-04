function [f,f_grad,ci,ci_grad,ce,ce_grad] = combinedFunction(X)


    x1 = X.x1;
    x2 = X.x2;
 

    f = 8*abs(x1^2 - x2) + (1 - x1)^2;

    % GRADIENT AT X
    % Compute the 2nd term
    f_grad.x1      = -2*(1-x1);
    f_grad.x2 = 0;
    % Add in the 1st term, where we must handle the sign due to the 
    % absolute value
    if x1^2 - x2 >= 0
      f_grad.x1    = f_grad.x1 + 8*2*x1;
      f_grad.x2    = f_grad.x2 + 8*(-1);
    else
      f_grad.x1    = f_grad.x1 - 8*2*x1;
      f_grad.x2    = f_grad.x2 + 8*(1);
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