function soln = runExample_matrix()

[A,B,C,p,m] = loadExample('ex4_data_n=200.mat');
nvar        = m*p;
x0          = zeros(nvar,1);
opts.x0     = x0;
opts.maxit  = 200;
stab_margin = 1;
feasibility_bias = false;
if feasibility_bias
    opts.steering_ineq_margin = inf;    % default is 1e-6
    opts.steering_c_viol = 0.9;         % default is 0.1
    opts.steering_c_mu = 0.1;           % default is 0.9
end



% key
inputVar = {'XX'};
% value: dimension. e.g., 2 by 2 => [2,2]
varDim = {[p,m]};
inputVarMap =  containers.Map(inputVar, varDim);

% calculate total number of scalar variables
nvar = 0;
for idx = 1:length(varDim)
    curDim = varDim(idx);
    nvar = nvar + curDim{1,1}(1)*curDim{1,1}(2);
end


combined_fn = @(x) matrixSolver(x,inputVarMap );

soln = granso(nvar,combined_fn,opts);
 

    
% % key
% inputVar = {'x1','x2'};
% % value: dimension. e.g., 2 by 2 => [2,2]
% varDim = {[1,1],[1,1]};
% inputVarMap =  containers.Map(inputVar, varDim);
% 
% % calculate total number of scalar variables
% nvar = 0;
% for idx = 1:length(varDim)
%     curDim = varDim(idx);
%     nvar = nvar + curDim{1,1}(1)*curDim{1,1}(2);
% end
% 
% 
% combined_fn = @(x) matrixSolver(x,inputVarMap);
% 
% soln = granso(nvar,combined_fn);



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