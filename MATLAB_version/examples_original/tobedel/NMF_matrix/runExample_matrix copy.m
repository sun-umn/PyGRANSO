function soln = runExample_matrix()

    nvar    = 8;    % the number of optimization variables is 2
%     eq_fn   = [];   % use [] when (in)equality constraints aren't present
%     ineq_fn   = [];

%     combined_fn = @(x) matrixSolver(x);
%     soln = granso(nvar, combined_fn);

 
    soln = granso(nvar, @matrixSolver);

%     % SET UP THE ANONYMOUS FUNCTION HANDLE AND OPTIMIZE
%     combined_fn = @(x) combinedFunction(A,B,C,stab_margin,x);
%     soln        = granso(nvar,combined_fn,opts);
    
%     soln    = granso(nvar,@objectiveFunction,@inequalityConstraint,eq_fn);
%     soln    = granso(nvar,@objectiveFunction,ineq_fn,eq_fn);
    % Alternatively, without the eq_fn variable:
    % soln    = granso(nvar,@objectiveFunction,@inequalityConstraint,[]);
    
end