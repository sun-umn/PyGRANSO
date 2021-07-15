function [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec] = mat2vec(x,var_dim_map,nvar,parameters)
%   mat2vec:
%       Transform user provided objective and constraint function from
%       matrix form to vector form, which is supported by GRANSO package.
%
%       combinedFunction.m need to be provided in the same folder.
%
%   USAGE:
%       [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec] = mat2vec(x,inputVarMap)
%
%   INPUT:
%       x           optimization variables
%                   real-valued column vector, nvar by 1, where nvar is the
%                   total number of scalar variables
%
%       var_dim_map        Map container used for storing input data, where key is
%       the input variable and value is the corresponding dimsions. e.g.,
%       key = {'U','V'} and dim = {[3,2],[4,2]}; or key = {'x1','x2'} and dim = {[1,1],[1,1]}
%
%       nvar          total number of scalar variables in x
%
%       parameters      struct of parameters used in the combinedFunction
%
%   OUTPUT:
%       f_vec           value of the objective function at x
%                       scalar real value
%
%       f_grad_vec      gradient of the objective function at x.
%                       real-valued column vector, n by 1, where n is the
%                       total number of scalar variables
%
%       ci_vec          values of the inequality constraints at x.
%                       real-valued column vector, m by 1, where the jth entry
%                       is the value of jth constraint
%
%       ci_grad_vec     gradient of the inequality constraint at x.
%                       real-valued matrix, n by m, where the jth column is the
%                       gradient of the jth constraint
%
%       ce_vec          (TODO) values of the equality constraints at x.
%                       real-valued column vector, m by 1, where the jth entry
%                       is the value of jth constraint
%
%
%       ce_grad_vec     (TODO) gradient of the equality constraint at x.
%                       real-valued matrix, n by m, where the jth column is the
%                       gradient of the jth constraint

% Handle missing arguments
if nargin == 3
    parameters = {}; % empty struct
end

% input variables (matrix form), e.g., {'U','V'};
var = keys(var_dim_map);
% corresponding dimensions (matrix form), e.g., {[3,2],[4,2]};
dim = values(var_dim_map);

%% reshape vector input x to matrix form X, e.g., X.U and X.V
curIdx = 0;
for idx = 1:length(var)
    % current variable, e.g., U
    tmpVar = var{1,idx};
    % corresponding dimension of the variable, e.g, 3 by 2
    tmpDim1 = dim{1,idx}(1);
    tmpDim2 = dim{1,idx}(2);
    % reshape vector input x in to matrix variables, e.g, X.U, X.V
    curIdx = curIdx + 1;
    X.(tmpVar) = reshape(x(curIdx:curIdx+tmpDim1*tmpDim2-1),tmpDim1,tmpDim2);
    curIdx = curIdx+tmpDim1*tmpDim2-1;
end

%% obtain objective and constraint function and their corresponding gradient

% matrix form functions
if isempty(parameters)
    [f,f_grad,ci,ci_grad,ce,ce_grad] = combinedFunction(X);
else
    [f,f_grad,ci,ci_grad,ce,ce_grad] = combinedFunction(X,parameters);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% obj function is a scalar form
f_vec = f;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% transform f_grad form matrix form to vector form
f_grad_vec = zeros(nvar,1);
curIdx = 0;
for idx = 1:length(var)
    % current variable, e.g., U
    tmpVar = var{1,idx};
    % corresponding dimension of the variable, e.g, 3 by 2
    tmpDim1 = dim{1,idx}(1);
    tmpDim2 = dim{1,idx}(2);
    curIdx = curIdx + 1;
    f_grad_vec(curIdx:curIdx+tmpDim1*tmpDim2-1) = f_grad.(tmpVar)(:);
    curIdx = curIdx+tmpDim1*tmpDim2-1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% number of constraints
nconstr = 0;
if ~(isempty(ci))
    arrConstr = fieldnames(ci);
    % get # of constraints
    for iconstr = 1: length(arrConstr)
        % current constraint, e.g., c1, c2
        tmpconstr = arrConstr{iconstr};
        constrMatrix = ci.(tmpconstr);
        
        nconstr = nconstr + length(constrMatrix(:));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % inquality constraints
    ci_vec = zeros(nconstr,1);
    curIdx = 0;
    for iconstr = 1: length(arrConstr)
        % current constraint, e.g., c1, c2
        tmpconstr = arrConstr{iconstr};
        constrMatrix = ci.(tmpconstr);
        curIdx = curIdx+1;
        ci_vec(curIdx:curIdx+length(constrMatrix(:))-1) = constrMatrix(:);
        curIdx = curIdx+length(constrMatrix(:))-1;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % gradient of inquality constraints
    ci_grad_vec = zeros(nvar,nconstr);
    % iterate column: constraints
    colIdx = 0;
    for iconstr = 1: length(arrConstr)
        % current constraint, e.g., c1, c2
        tmpconstr = arrConstr{iconstr};
        constrMatrix = ci.(tmpconstr);
        rowIdx = 0;
        colIdx = colIdx+1;
        % iterate row: variables
        for idx = 1:length(keys(var_dim_map))
            % current variable, e.g., U
            tmpVar = var{1,idx};
            ciGradMatrix = ci_grad.(tmpconstr).(tmpVar);
            % corresponding dimension of the variable, e.g, 3 by 2
            tmpDim1 = dim{1,idx}(1);
            tmpDim2 = dim{1,idx}(2);
            rowIdx = rowIdx + 1;
            ci_grad_vec(rowIdx:rowIdx+tmpDim1*tmpDim2-1,colIdx:colIdx+length(constrMatrix(:))-1) = ciGradMatrix;
            rowIdx = rowIdx +tmpDim1*tmpDim2-1;
        end
        colIdx = colIdx+length(constrMatrix(:))-1;
    end
    
else
    ci_vec =[];
    ci_grad_vec = [];
end
%% TODO: equality constraints
ce_vec = ce;
ce_grad_vec = ce_grad;

end