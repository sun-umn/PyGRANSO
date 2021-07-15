function H_obj = bfgsHessianInverseLimitedMem(  H0,                 ...
                                                scaleH0,            ...
                                                fixed_scaling,      ...
                                                nvec,               ...
                                                restart_data        )
%   bfgsHessianInverseLimitedMem:
%       An object that maintains and updates an L-BFGS approximation to the 
%       inverse Hessian.
%
%   INPUT:
%       HO                  [ positive definite matrix ]
%           Initial inverse Hessian approximation to use in L-BFGS update
%           formula.
% 
%       scaleH0             [ logical ]
%           Logical indicating whether or not to employ scaling to H0.
%
%       fixed_scaling       [ logical ]
%           When fixed_scaling is true (and scaleH0 is also true), the
%           first calculated value of the scaling parameter will be used
%           for all subsequent updates.  Otherwise, a new value of the
%           scaling parameter will be calculated on each update.
%
%       nvec                [ positive integer ]
%           number of vectors to store for L-BFGS.
%
%       restart_data        [ struct ]
%           Struct of data to warm-start L-BFGS from previous information:
%           .S          n by m real-valued finite matrix of s vectors
%           .Y          n by m real-valued finite matrix of y vectors
%           .rho        length m real-valued row vector of 1/sty values
%           .gamma      real finite value
%           H_obj.getState() returns a struct containing this data.
%                   
%   OUTPUT:
%       H_obj               [ a struct ] 
%           A struct containing function handles for maintaining an L-BFGS
%           inverse Hessian approximation.
% 
%           skipped = H_obj.update(s,y,sty,damped)
%           Attempt an L-BFGS update.
%           INPUT:
%               s           BFGS vector: s = x_{k+1} - x_k = alpha*d
%               y           BFGS vector: y = g_{k_1} - g_k
%               sty         s.'*y
%               damped      logical: whether or not damping was applied to
%                           y to ensure that sty > 0 
%           OUTPUT:
%               skipped     an integer indicating the status:
%                               skipped = 0:    successful L-BFGS update
%                               skipped = 1:    update but without scaling
%                               skipped = 2:    failed, s.'*y <= 0
%                               skipped = 3:    failed, contained nans/infs
% 
%           Hx = H_obj.applyH(x) 
%           Multiply matrix/vector x by H, the current L-BFGS inverse
%           Hessian approximation
%
%           H = H_obj.getState()
%           Returns a struct with current L-BFGS state:
%               .S          n by m real-valued finite matrix of s vectors
%               .Y          n by m real-valued finite matrix of y vectors
%               .rho        length m real-valued row vector of 1/sty values
%               .gamma      real finite value
%                   
%           data = H_obj.getCounts()
%           Returns a struct of data with counts of the following:
%               requests            total requests to update
%               updates             total accepted updates
%               damped_requests     request was a damped update
%               damped_updates      update was damped
%               sty_fails           sty <= 0 prevented update
%               infnan_fail         inf/nan in result prevented update
%               scaling_skips       numerical issue prevent scaling 
%
%
%   If you publish work that uses or refers to GRANSO, please cite the 
%   following paper: 
%
%   [1] Frank E. Curtis, Tim Mitchell, and Michael L. Overton 
%       A BFGS-SQP method for nonsmooth, nonconvex, constrained 
%       optimization and its evaluation using relative minimization 
%       profiles, Optimization Methods and Software, 2016.
%       Available at https://dx.doi.org/10.1080/10556788.2016.1208749
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   bfgsHessianInverseLimitedMem.m introduced in GRANSO Version 1.5.
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

    n                   = size(H0,1);
    S                   = zeros(n,nvec);
    Y                   = zeros(n,nvec);
    rho                 = zeros(1,nvec);
    gamma               = 1;
    count               = 0;
    update_gamma        = scaleH0;
    
    requests            = 0;
    updates             = 0;
    damped_requests     = 0;
    damped_updates      = 0;
    scale_fails         = 0;
    sty_fails           = 0;
    infnan_fails        = 0;
    
    if nargin > 4 && isstruct(restart_data)
        cols = size(restart_data.S,2);
        if cols >= nvec
            S           = restart_data.S(:,1:nvec);
            Y           = restart_data.Y(:,1:nvec);
            rho         = restart_data.rho(1:nvec);
            count       = nvec;
        else
            S(:,1:cols) = restart_data.S;
            Y(:,1:cols) = restart_data.Y;
            rho(1:cols) = restart_data.rho;
            count       = cols;
        end
        gamma           = restart_data.gamma;
    end
   
    H_obj   = struct(   'update',       @update,    ...
                        'applyH',       @applyH,    ...
                        'applyHreg',    @applyH,    ...
                        'getState',     @getState,  ...
                        'getCounts',    @getCounts  );
                       
    function skipped = update(s,y,sty,damped)
  
        requests            = requests + 1;
        skipped             = 0;
        
        damped              = nargin > 3 && damped;
        if damped
            damped_requests = damped_requests + 1;
        end
        
        % sty > 0 should hold in theory but it can fail due to rounding
        % errors, even with if BFGS damping is applied. 
        if sty <= 0
            skipped         = 2;
            sty_fails       = sty_fails + 1;
            return
        end
        
        % We should also check that s and y only have finite entries.
        if  any(isinf(s) | isnan(s) | isinf(y) | isnan(y)) 
            skipped         = 3;
            infnan_fails    = infnan_fails + 1;
            return
        end
        
        rho_new = 1/(sty);  % this will be finite since sty > 0
        
        if update_gamma 
            gamma_new       = 1 / (rho_new * (y.' * y));
            if isinf(gamma_new) || isnan(gamma_new)
                % can still apply update with scaling disabled and/or
                % previous value of scaling parameter.  We'll update the
                % latter approach.
                skipped     = 1;
                scale_fails = scale_fails + 1;
            else
                gamma       = gamma_new;       
                if fixed_scaling
                    update_gamma = false;
                end
            end
        end
        
        if count < nvec
            count           = count + 1;
            S(:,count)      = s;
            Y(:,count)      = y;
            rho(count)      = rho_new;    
        else
            S               = [ S(:,2:end) s ];
            Y               = [ Y(:,2:end) y ];
            rho             = [ rho(2:end) rho_new ];
        end
        
        updates             = updates + 1;
        if damped
            damped_updates  = damped_updates + 1;
        end
    end

    function r = applyH(q)
             
        % q might be a matrix, not just a vector, so we want to apply
        % multiplication to all columns of q
        cols    = size(q,2);
        alpha   = zeros(count,cols);
      
        % Now apply first updates to the columns of q
        for j = count:-1:1
            alpha(j,:)  = rho(j) * (S(:,j).'*q);
            y           = Y(:,j);
            for k = 1:cols
                q(:,k)  = q(:,k) - alpha(j,k) * y;
            end
        end
        
        % Apply the sparse matvec and (possible) scaling 
        r = gamma*(H0*q);
        
        % Finally apply updates to the columns of r
        for j = 1:count
            beta        = rho(j) * (Y(:,j)'*r);
            s           = S(:,j);
            for k = 1:cols
                r(:,k)  = r(:,k) + (alpha(j,k) - beta(k)) * s;
            end
        end   
    end  

    function data = getState()
        data = struct(  'S',        S(:,1:count),       ...
                        'Y',        Y(:,1:count),       ...
                        'rho',      rho(1:count),       ...
                        'gamma',    gamma               );
    end

    function counts = getCounts()
        counts = struct(    'requests',         requests,           ...
                            'updates',          updates,            ...
                            'damped_requests',  damped_requests,    ...
                            'damped_updates',   damped_updates,     ...
                            'scaling_skips',    scale_fails,        ...
                            'sty_fails',        sty_fails,          ...
                            'infnan_fails',     infnan_fails        );     
    end
end