function soln = runExample()
%   runExample: (examples/ex4)
%       Run GRANSO on an eigenvalue optimization problem of a 
%       static-output-feedback (SOF) plant:
%       
%           M = A + BXC,
%
%       where A,B,C are all fixed real-valued matrices
%           - A is n by n 
%           - B is n by p
%           - C is m by n
%       and X is a real-valued m by p matrix of the optimization variables.
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
%       This tutorial example shows:   
%
%           - how to use GRANSO's "combined" format to easily share 
%             expensive computed results that are needed in both the
%             objective and inequality constraints, namely the repeated
%             eigenvalue decomposition of a 200 by 200 matrix
%
%           - how to tune some of GRANSO's advanced parameter which balance
%             the goals of feasibility and objective minimization.  See the
%             feasibility_bias variable, which when true, sets some of 
%             GRANSO's advanced parameters
%
%           - how to use GRANSO's limited-memory mode.
%       
%       The behavior of this example can be changed by adjusting the values
%       of following internal variables:
%           - opts.maxit            [positive integer]
%           - opts.limited_mem_size [empty or nonnegative integer]
%           - stab_margin           [nonnegative real-valued scalar]
%           - feasibility_bias      [logical]
%         
%   USAGE:
%       soln = runExample();
% 
%   INPUT: [none]
%   
%   OUTPUT:
%       soln        GRANSO's output struct
%
%   For additional nonsmooth constrained optimization problems, using SOF 
%   plants, see:
%
%   PSARNOT: (Pseudo)Spectral Abscissa|Radius Nonsmooth Optimization Test
%   http://www.timmitchell.com/software/PSARNOT
% 
%   See also combinedFunction, gransoOptionsAdvanced.
%
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   examples/ex4/runExample.m introduced in GRANSO Version 1.5.
%
% =========================================================================
% |  GRANSO: GRadient-based Algorithm for Non-Smooth Optimization         |
% |  Copyright (C) 2016 Tim Mitchell                                      |
% |                                                                       |
% |  This file is part of GRANSO.                                         |
% |                                                                       |
% |  GRANSO is free software: you can redistribute it and/or modify       |
% |  it under the terms of the GNU Affero General Public License as       |
% |  published by the Free Software Foundation, either version 3 of       |
% |  the License, or (at your option) any later version.                  |
% |                                                                       |
% |  GRANSO is distributed in the hope that it will be useful,            |
% |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
% |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
% |  GNU Affero General Public License for more details.                  |
% |                                                                       |
% |  You should have received a copy of the GNU Affero General Public     |
% |  License along with this program.  If not, see                        |
% |  <http://www.gnu.org/licenses/agpl.html>.                             |
% =========================================================================

    % LOAD DATA FOR PROBLEM AND SET INITIAL POINT
    % Don't alter the following 4 lines
    [A,B,C,p,m] = loadExample('ex4_data_n=200.mat');
    nvar        = m*p;
    x0          = zeros(nvar,1);
    opts.x0     = x0;
    
    
    % ADJUSTABLE PARAMETERS
    % This should probably be at least 100
    opts.maxit  = 200;  
    
    % Limited-memory mode is generally not recommended for nonsmooth
    % problems, such as this one, but it can nonetheless enabled if
    % desired/necessary.  opts.limited_mem_size == 0 is off, that is, 
    % limited-memory mode is disabled.
    % Note that this example has 200 variables. 
%     opts.limited_mem_size = 40;
    
    % This should be a nonnegative value representing how far into the left
    % half-plane the eigenvalues of M = A + BXC must be in order to have 
    % satisfied the stability constraint.  e.g. when equal to 1, the
    % spectral abscissa of M must be no more than -1.
    stab_margin = 1;    
    
    % We can also tune GRANSO to more aggressively favor satisfying
    % feasibility over minimizing the objective.  Set feasibility_bias to
    % true to adjust the following three steering parameters away from
    % their default values.  For more details on these parameters, type
    % >> help gransoOptionsAdvanced
    feasibility_bias = false;
    if feasibility_bias
        opts.steering_ineq_margin = inf;    % default is 1e-6
        opts.steering_c_viol = 0.9;         % default is 0.1
        opts.steering_c_mu = 0.1;           % default is 0.9
    end
    
    % In my testing, with default parameters, GRANSO will first obtain a
    % feasible solution at iter = 98 and will reduce the objective to
    % 11.156 by the time it attains max iteration count of 200.
    
    % With feasibility_bias = true, in my testing, GRANSO will obtain its
    % first feasible solution earlier, at iter = 62, but it will ultimately
    % have reduced the objective value less, only to 11.749, by the end of
    % its 200 maximum allowed iterations.
    
  
    % NO NEED TO CHANGE ANYTHING BELOW THIS LINE 
    
    % FOR PLOTTING THE SPECTRUM
    org_color   = [255 174 54]/255;
    opt_color   = [57 71 198]/255;
    
    clf
    plotSpectrum(opts.x0,org_color);
    hold on
    fprintf('Initial spectrum represented by orange dots.\n');
    fprintf('Press any key to begin A+BXC optimization.\n');
    pause
    
    
    % SET UP THE ANONYMOUS FUNCTION HANDLE AND OPTIMIZE
    combined_fn = @(x) combinedFunction(A,B,C,stab_margin,x);
    soln        = granso(nvar,combined_fn,opts);
    
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


