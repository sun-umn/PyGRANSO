clc
clear

soln = runExample();

% soln = ans;

V = soln.final.x(1:10);
W = soln.final.x(11:100);
W = reshape(W,[9,10]);

data = readtable('breast-cancer-wisconsin.csv');
[N,p] = size(data);
p = p - 2; % delete index and y

data.Var7 = str2double (data.Var7);

y = data(:,p+2);
x = data(:,2:p+1);


y = y{:,:};
x = x{:,:};
x(isnan(x)) = 1;

Z = Sigmoid(x * W);
y_pred = Sigmoid(Z * V); % pred

y_pred(y_pred>=0.5) = 4;
y_pred(y_pred<0.5) = 2;

acc = sum(y==y_pred)/699;

disp(acc)

function Y = Sigmoid(X)
    Y = power(1+exp(-X), -1);
end