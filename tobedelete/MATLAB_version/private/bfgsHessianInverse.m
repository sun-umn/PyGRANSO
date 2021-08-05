function H_obj = bfgsHessianInverse(H,scaleH0,varargin)
%   bfgsHessianInverse:
%       An object that maintains and updates a BFGS approximation to the 
%       inverse Hessian.
%
%   INPUT:
%       H               [ positive definite matrix ]
%           Initial inverse Hessian approximation
% 
%       scaleH0         [ logical ]
%           Logical indicating whether or not to employ scaling on the
%           first BFGS update.  Scaling is never applied on any subsequent
%           update.
%
%   OUTPUT:
%       H_obj           [ a struct ] 
%           A struct containing function handles for maintaining a BFGS
%           inverse Hessian approximation.
% 
%           skipped = H_obj.update(s,y,sty,damped)
%           Attempt a BFGS update.
%           INPUT:
%               s           BFGS vector: s = x_{k+1} - x_k = alpha*d
%               y           BFGS vector: y = g_{k_1} - g_k
%               sty         s.'*y
%               damped      logical: whether or not damping was applied to
%                           y to ensure that sty > 0 
%           OUTPUT:
%               skipped     an integer indicating the status:
%                               skipped = 0:    successful BFGS update
%                               skipped = 1:    update but without scaling
%                               skipped = 2:    failed, s.'*y <= 0
%                               skipped = 3:    failed, contained nans/infs
% 
%           Hx = H_obj.applyH(x)     
%           Multiply matrix/vector x by H, the current BFGS inverse Hessian
%           approximation
%
%           H = H_obj.getState()
%           Return H, the current BFGS inverse Hessian approximation
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
%   bfgsHessianInverse.m introduced in GRANSO Version 1.0.
%
% =========================================================================
% |  bfgsHessianInverse.m                                                 |
% |  Copyright (C) 2016 James Burke, Adrian Lewis, Tim Mitchell, and      |
% |  Michael Overton                                                      |
% |                                                                       |
% |  Parts of this routine (this single file) are taken from the HANSO    |
% |  software package, which is licensed under the GPL v3.  As such, this |
% |  single routine is also licensed under the GPL v3.  However, note     |
% |  that this is an exceptional case; GRANSO and most of its subroutines |
% |  are licensed under the AGPL v3.                                      |
% |                                                                       |
% |  This routine (this single file) is free software: you can            |
% |  redistribute it and/or modify it under the terms of the GNU General  |
% |  Public License as published by the Free Software Foundation, either  |
% |  version 3 of the License, or (at your option) any later version.     |
% |                                                                       |
% |  This routine is distributed in the hope that it will be useful,      |
% |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
% |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    |
% |  General Public License for more details.                             |
% |                                                                       |
% |  You should have received a copy of the GNU General Public License    |
% |  along with this program.  If not, see                                |
% |  <http://www.gnu.org/licenses/gpl.html>.                              |
% =========================================================================
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
    
    requests        = 0;
    updates         = 0;
    damped_requests = 0;
    damped_updates  = 0;
    scale_fails     = 0;
    sty_fails       = 0;
    infnan_fails    = 0;
        
    H_obj   = struct(   'update',       @update,    ...
                        'applyH',       @applyH,    ...
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
        % errors, even if BFGS damping is applied. 
        if sty <= 0
            skipped         = 2;
            sty_fails       = sty_fails + 1;
            return
        end
        
        % sty > 0 so attempt BFGS update
        if scaleH0
            % for full BFGS, Nocedal and Wright recommend scaling I
            % before the first update only
            gamma           = sty/(y'*y);
            if isinf(gamma) || isnan(gamma)
                skipped     = 1;
                scale_fails = scale_fails + 1;
            else
                H           = gamma*H;
            end
            scaleH0         = false; % only allow first update to be scaled
        end
        
        % for formula, see Nocedal and Wright's book
        %M = I - rho*s*y', H = M*H*M' + rho*s*s', so we have
        %H = H - rho*s*y'*H - rho*H*y*s' + rho^2*s*y'*H*y*s' + rho*s*s'
        % note that the last two terms combine: (rho^2*y'Hy + rho)ss'
        rho                 = 1/sty;
        Hy                  = H*y;
        rhoHyst             = (rho*Hy)*s';
        
        % old version: update may not be symmetric because of rounding
        % H = H - rhoHyst' - rhoHyst + rho*s*(y'*rhoHyst) + rho*s*s';
        % new in HANSO version 2.02: make H explicitly symmetric
        % also saves one outer product
        % in practice, makes little difference, except H=H' exactly
        
        % ytHy could be < 0 if H not numerically pos def
        ytHy                = y'*Hy;
        sstfactor           = max([rho*rho*ytHy + rho,  0]);
        sscaled             = sqrt(sstfactor)*s;
        H_new               = H - (rhoHyst' + rhoHyst) + sscaled*sscaled';
        
        % only update H if H_new doesn't contain any infs or nans
        H_vec               = H_new(:);
        if any(isinf(H_vec) | isnan(H_vec))
            skipped         = 3;
            infnan_fails    = infnan_fails + 1;
        else
            H               = H_new;
            updates         = updates + 1;
            if damped 
                damped_updates = damped_updates + 1;
            end
        end
        
        % An aside, for an alternative way of doing the BFGS update:
        % Add the update terms together first: does not seem to make
        % significant difference
        %   update = sscaled*sscaled' - (rhoHyst' + rhoHyst);
        %   H = H + update;
    end

    function r = applyH(q)
        r = H*q;
    end

    function H_out = getState()
        H_out = H;
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