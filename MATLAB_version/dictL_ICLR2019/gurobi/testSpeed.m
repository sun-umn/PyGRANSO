n = 500;   % dimension
m = round(10*n^2);    % number of measurements

Y= ones(n,m);
q = ones(n,1);


fprintf("f_grad.q = YTq;")
tic
output = Y* Y'*q;
toc