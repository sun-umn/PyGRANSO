function [ci,ci_grad] = inequalityConstraint_matrix(X)

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
    
    [ci.c1,indx]   = max(real(d));
    indx        = indx(1); 
    lambda      = d(indx);
    % account for the stability margin in the inequality constraint
    ci.c1          = ci.c1 + stability_margin;
     

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
 

%     x1 = X.x1;
%     x2 = X.x2;
% 
%     ci.c1 = sqrt(2)*x1-1;
%     ci.c2 = 2*x2-1;
%     
%     
%     % # of constr b # of vars of U and V
%     ci_grad.c1.x1 = sqrt(2);
%     ci_grad.c1.x2 = 0;
%     
%     ci_grad.c2.x1 = 0;
%     ci_grad.c2.x2 = 2;

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