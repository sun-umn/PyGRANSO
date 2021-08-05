function [f,f_grad,ci,ci_grad,ce,ce_grad] = combinedFunction(X,parameters)
    q = X.q;

    Y = parameters.Y;
    m = parameters.m;
    
    f = 1/m*norm(q'*Y, 1);
    f_grad.q = 1/m*Y*sign(Y'*q);
    
    ci = [];
    ci_grad = [];
    
    ce.c1 = q'*q - 1;
    ce_grad.c1.q = 2*q;
    
          
end