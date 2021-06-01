clc
clear

soln = runExample_matrix();

U = reshape(soln.final.x(1:4),2,2);

V = reshape(soln.final.x(5:8),2,2);

U*V'