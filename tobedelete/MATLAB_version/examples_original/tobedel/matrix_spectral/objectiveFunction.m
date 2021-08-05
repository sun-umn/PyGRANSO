function [f,f_grad] = objectiveFunction_matrix(X)
    
    [A,B,C,p,m] = loadExample('ex4_data_n=200.mat');
    stability_margin = 1;

    p           = size(B,2);
    m           = size(C,1);
    XX           = X.XX;
    M           = A + B*XX*C;
    [V,D]       = eig(M);
    d           = diag(D);     
    [VL,D_conj] = eig(M');
    dl          = conj(diag(D_conj));
    [mi,indx]   = max(imag(d));
    indx        = indx(1);
    lambda      = d(indx);
    f           = mi;

    x           = V(:,indx);
    % Now find the matching left eigenvector for lambda
    [~,indx]    = min(abs(dl - lambda));
    y           = VL(:,indx);
    Bty         = B.'*y;
    Cx          = C*x;
    % Gradient of inner product with respect to A 
    f_grad.XX      = imag((conj(Bty)*Cx.')/(y'*x));

    

%     x1 = X.x1;
%     x2 = X.x2;
%  
% 
%     f = 8*abs(x1^2 - x2) + (1 - x1)^2;
% 
%     % GRADIENT AT X
%     % Compute the 2nd term
%     f_grad.x1      = -2*(1-x1);
%     f_grad.x2 = 0;
%     % Add in the 1st term, where we must handle the sign due to the 
%     % absolute value
%     if x1^2 - x2 >= 0
%       f_grad.x1    = f_grad.x1 + 8*2*x1;
%       f_grad.x2    = f_grad.x2 + 8*(-1);
%     else
%       f_grad.x1    = f_grad.x1 - 8*2*x1;
%       f_grad.x2    = f_grad.x2 + 8*(1);
%     end
    
%     U = X.U;
%     V = X.V;
%  
%     D = [1 1;1 1;1 1];
%     
%     f = .5*norm(D - U*V', 'fro')^2;
% 
%     f_grad.U = -(D - U*V') * V; 
%     f_grad.V = -(D - U*V')' * U; 


%     U = X.U;
%     V = X.V;
%  
%     D = [1 1; 1 1];
%     
%     f = .5*norm(D - U*V', 'fro')^2;
% 
%     f_grad.U = -(D - U*V') * V; 
%     f_grad.V = -(D - U*V')' * U; 
    
    
%     f = 0.5 * ( (1-x(1)*x(5)-x(2)*x(7))^2 + (1-x(1)*x(6)-x(2)*x(8))^2 ...
%         + (1-x(3)*x(5)-x(4)*x(7))^2 + (1-x(3)*x(6)-x(4)*x(8))^2   );    
%     f_grad = [-x(5)*(1-x(1)*x(5)-x(2)*x(7)) - x(6)*(1-x(1)*x(6)-x(2)*x(8))   ;
%                -x(7)*(1-x(1)*x(5)-x(2)*x(7)) - x(8)*(1-x(1)*x(6)-x(2)*x(8)) ;
%               - x(5)*(1-x(3)*x(5)-x(4)*x(7)) - x(6)*(1-x(3)*x(6)-x(4)*x(8))  ;
%               - x(7)*(1-x(3)*x(5)-x(4)*x(7)) - x(8)*(1-x(3)*x(6)-x(4)*x(8))  ;
%              -x(1)*(1-x(1)*x(5)-x(2)*x(7)) - x(3)*(1-x(3)*x(5)-x(4)*x(7))   ;
%               - x(1)*(1-x(1)*x(6)-x(2)*x(8)) - x(3)*(1-x(3)*x(6)-x(4)*x(8)) ;
%               -x(2)*(1-x(1)*x(5)-x(2)*x(7))- x(4)*(1-x(3)*x(5)-x(4)*x(7))   ;
%               - x(2)*(1-x(1)*x(6)-x(2)*x(8))- x(4)*(1-x(3)*x(6)-x(4)*x(8)) ];

          
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