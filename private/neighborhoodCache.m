function get_neighborhood_fn = neighborhoodCache(max_size,radius)
%   neighborhoodCache:
%       Maintains a cache of objects (up to max_size) that all have an
%       associated n-dimensional point.  The function can return the
%       objects whose points are within a fixed distance (radius) of a
%       query point object. (which is then also added to the cache).
%
%   USAGE:
%       get_nbd_fn = neighborhoodCache(max_size,radius);
%       [nbd_x,nbd_objects,distances_computed] = get_nbd_fn(x,object);
%
%   INPUT:
%       x                   an n-dimensional point (n must be the same for
%                           all calls to get_nbd_fn
%
%       objects             object to cache for point x
%       
%   OUTPUT:
%       nbd_x               n by l array of l n-dimensional points in the
%                           neighborhood of x (which also includes x)
%   
%       nbd_objects         cell array of length l of l objects associated 
%                           with the points in nbd_x
% 
%       distances_computed  number of pairwise point distances from x 
%                           computed 
%
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   neighborhoodCache.m introduced in GRANSO Version 1.0.
%
% =========================================================================
% |  neighborhoodCache.m                                                  |
% |  Copyright (C) 2016 Tim Mitchell                                      |
% |                                                                       |
% |  This file is originally from URTM.                                   |
% |                                                                       |
% |  URTM is free software: you can redistribute it and/or modify         |
% |  it under the terms of the GNU Affero General Public License as       |
% |  published by the Free Software Foundation, either version 3 of       |
% |  the License, or (at your option) any later version.                  |
% |                                                                       |
% |  URTM is distributed in the hope that it will be useful,              |
% |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
% |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
% |  GNU Affero General Public License for more details.                  |
% |                                                                       |
% |  You should have received a copy of the GNU Affero General Public     |
% |  License along with this program.  If not, see                        |
% |  <http://www.gnu.org/licenses/agpl.html>.                             |
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
    
    % by default, inf distances indicate empty slots for samples
    distances       = inf(1,max_size);
    samples         = [];
    data            = cell(1,max_size);
  
    n               = 0;
    last_added_ind  = 0;

    get_neighborhood_fn = @getCachedNeighborhoodAbout;
    
    function [n_x,n_data,computed] = getCachedNeighborhoodAbout(x,x_data)
        if last_added_ind == 0
            n               = 1;
            last_added_ind  = 1;
            distances(1)    = 0;
            samples         = zeros(length(x),max_size);
            samples(:,1)    = x;
            data{1}         = x_data;
            computed        = 0;
        else
            % Calculate the distance from the new sample point x to the 
            % most recent;y added sample already in the cache 
            dist_to_last_added = norm(x - samples(:,last_added_ind));
            distances(last_added_ind) = 0; % will be set exactly below
            
            % Overestimate the distances from the new sample point x to all 
            % the other sample points by applying the triangle inequality 
            % to dist_to_most_recent and the previously computed/estimated 
            % distances between samples{last_index} and all the remaining
            % points. 
            % Note: distance(last_added_index) will be dist_to_last_added.
            distances(1:n)      = distances(1:n) + dist_to_last_added;
            
            % Only the (over)estimated distances which are greater than the 
            % allowed radius will need to be computed exactly.
            indx                = distances > radius & ~isinf(distances);
            computed            = sum(indx);
            distances(indx)     = sqrt(sum(samples(:,indx).^2));
           
            if n < max_size
                % add x and x_data to next free slot
                n               = n + 1;
                last_added_ind  = n;
                distances(n)    = 0;
                samples(:,n)    = x;
                data{n}         = x_data;
            else
                % no free slot available - overwrite oldest sample
                oldest_ind              = mod(last_added_ind,max_size) + 1;
                distances(oldest_ind)   = 0;
                samples(:,oldest_ind)   = x;
                data{oldest_ind}        = x_data;
                last_added_ind          = oldest_ind;
            end
        end
        % selection to return
        indx            = distances <= radius;
        n_x             = samples(:,indx);
        n_data          = data(indx);
    end
end