clc
clear

soln = runExample_matrix();

U = reshape(soln.final.x(1:6),3,2);

V = reshape(soln.final.x(7:10),2,2);

U*V'