function soln = runExample_matrix()

    
 
    
% key
inputVar = {'x1','x2'};
% value: dimension. e.g., 2 by 2 => [2,2]
varDim = {[1,1],[1,1]};
inputVarMap =  containers.Map(inputVar, varDim);

% calculate total number of scalar variables
nvar = 0;
for idx = 1:length(varDim)
    curDim = varDim(idx);
    nvar = nvar + curDim{1,1}(1)*curDim{1,1}(2);
end


combined_fn = @(x) matrixSolver(x,inputVarMap);

soln = granso(nvar,combined_fn);



%     % key
%     inputVar = {'U','V'}; 
%     % value: dimension. e.g., 2 by 2 => [2,2]
%     varDim = {[3,2],[2,2]};
%     inputVarMap =  containers.Map(inputVar, varDim);
%     
%     % calculate total number of scalar variables
%     nvar = 0;
%     for idx = 1:length(varDim)
%         curDim = varDim(idx);
%         nvar = nvar + curDim{1,1}(1)*curDim{1,1}(2);
%     end
%     
%     
%     combined_fn = @(x) matrixSolver(x,inputVarMap);
%     
%     soln = granso(nvar,combined_fn);
    
    
%     soln = granso(@matrixSolver);

%     % SET UP THE ANONYMOUS FUNCTION HANDLE AND OPTIMIZE
%     combined_fn = @(x) combinedFunction(A,B,C,stab_margin,x);
%     soln        = granso(nvar,combined_fn,opts);
    
%     soln    = granso(nvar,@objectiveFunction,@inequalityConstraint,eq_fn);
%     soln    = granso(nvar,@objectiveFunction,ineq_fn,eq_fn);
    % Alternatively, without the eq_fn variable:
    % soln    = granso(nvar,@objectiveFunction,@inequalityConstraint,[]);
    
end