function soln = runExample()
%   runExample: (example_mat/ex4)
%       Run GRANSO on an eigenvalue optimization problem of a 
%       static-output-feedback (SOF) plant:
%       
%           M = A + BXC,
% 
%       where A,B,C are all fixed real-valued matrices
%           - A is n by n 
%           - B is n by p
%           - C is m by n
%       and X is a real-valued p by m matrix of the optimization variables.
% 
%       The specific instance loaded when running this example has the
%       following properties:
%           - A,B,C were all generated via randn()
%           - n = 200
%           - p = 10
%           - m = 20
% 
%       The objective is to minimize the maximum of the imaginary parts of
%       the eigenvalues of M.  In other words, we want to restrict the
%       spectrum of M to be contained in the smallest strip as possible
%       centered on the x-axis (since the spectrum of M is symmetric with
%       respect to the x-axis).
% 
%       The (inequality) constraint is that the system must be
%       asymptotically stable, subject to a specified stability margin.  In
%       other words, if the stability margin is 1, the spectral abscissa 
%       must be at most -1.
% 
%       Read this source code.
% 
%   USAGE:
%       soln = runExample();
% 
%   INPUT: [none]
% 
%   OUTPUT:
%       soln        GRANSO's output struct


[A,B,C,p,m] = loadExample('ex4_data_n=200.mat');
nvar        = m*p;
x0          = zeros(nvar,1);
opts.x0     = x0;
opts.maxit  = 200; 
feasibility_bias = false;
if feasibility_bias
    opts.steering_ineq_margin = inf;    % default is 1e-6
    opts.steering_c_viol = 0.9;         % default is 0.1
    opts.steering_c_mu = 0.1;           % default is 0.9
end
stab_margin = 1; 

%% specify input variables 
% key: input variables
inputVar = {'XX'};
% value: dimension. e.g., 3 by 2 => [3,2]
varDim = {[p,m]};
inputVarMap =  containers.Map(inputVar, varDim);

% calculate total number of scalar variables
nvar = 0;
for idx = 1:length(varDim)
    curDim = varDim(idx);
    nvar = nvar + curDim{1,1}(1)*curDim{1,1}(2);
end




% %% FOR PLOTTING THE SPECTRUM
% org_color   = [255 174 54]/255;
% opt_color   = [57 71 198]/255;
% 
% clf
% plotSpectrum(opts.x0,org_color);
% hold on
% fprintf('Initial spectrum represented by orange dots.\n');
% fprintf('Press any key to begin A+BXC optimization.\n');
% pause


% SET UP THE ANONYMOUS FUNCTION HANDLE AND OPTIMIZE
%% call mat2vec to enable GRANSO using matrix input
% opts.quadprog_opts.QPsolver = 'qpalm';
opts.quadprog_opts.QPsolver = 'quadprog';
% opts.quadprog_opts.QPsolver = 'gurobi';
parameters.A = A;
parameters.B = B;
parameters.C = C;

tic
combined_fn = @(x) mat2vec(x,inputVarMap, nvar,parameters );
soln = granso(nvar,combined_fn,opts);
toc 

% PLOT THE OPTIMIZED SPECTRUM
hold on
[~,mi]      = plotSpectrum(soln.final.x,opt_color);
x_lim       = xlim();
y_lim       = ylim();
plot(x_lim,mi*[1 1],'--','Color',0.5*[1 1 1]);
plot(x_lim,-mi*[1 1],'--','Color',0.5*[1 1 1]);
plot(-stab_margin*[1 1],y_lim,'--','Color',0.5*[1 1 1]);

fprintf('\n\nInitial spectrum represented by orange dots.\n');
fprintf('Optimized spectrum represented by blue dots.\n');
fprintf('Stability margin represented by dashed vertical line.\n');
fprintf('The minimized band containing the spectrum represented by dashed horizontal lines.\n');

% NESTED HELPER FUNCTION FOR PLOTTING THE SPECTRUM
    function [absc,mi] = plotSpectrum(x,color)
        X       = reshape(x,p,m);
        d       = eig(A+B*X*C);
        absc    = max(real(d));
        mi      = max(imag(d));
        plot(real(d),imag(d),'.','MarkerSize',10,'Color',color);
    end
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