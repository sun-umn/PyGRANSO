clc
clear

soln = runExample();

U = reshape(soln.final.x(1:6),3,2);

V = reshape(soln.final.x(7:14),4,2);

U*V'