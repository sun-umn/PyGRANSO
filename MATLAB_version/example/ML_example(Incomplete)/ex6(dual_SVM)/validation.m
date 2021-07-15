clc
clear

soln = runExample();

% soln = ans;

alpha = soln.final.x;

data = readtable('breast-cancer-wisconsin.csv');
[N,p] = size(data);
p = p - 2; % delete index and y

data.Var7 = str2double (data.Var7);

y = data(:,p+2);
x = data(:,2:p+1);


y = y{:,:};
x = x{:,:};
x(isnan(x)) = 1;

w = sum(y .* alpha .* x); % p by 1

pred = x*w';

pred(pred>=3) = 4;
pred(pred<3) = 2;

acc = sum(y==pred)/699;

disp(acc)