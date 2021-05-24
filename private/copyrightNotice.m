function lines = copyrightNotice()
%   copyrightNotice:
%       This file returns a cell array of strings for printing GRANSO's
%       name, version, and copyright information.  Each string specifies
%       one line. The file should be modified accordingly when ever the
%       version number is increased.
%
%
%   If you publish work that uses or refers to GRANSO, please cite the
%   following paper:
%
%   [1] Frank E. Curtis, Tim Mitchell, and Michael L. Overton
%       A BFGS-SQP method for nonsmooth, nonconvex, constrained
%       optimization and its evaluation using relative minimization
%       profiles, Optimization Methods and Software, 32(1):148-181, 2017.
%       Available at https://dx.doi.org/10.1080/10556788.2016.1208749
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   nameVersionCopyrightMsg.m introduced in GRANSO Version 1.5.1.
%   Changed to copyrightNotice.m in v1.6
%
% =========================================================================
% |  GRANSO: GRadient-based Algorithm for Non-Smooth Optimization         |
% |  Copyright (C) 2016-2020 Tim Mitchell                                 |
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

lines = {
    'GRANSO: GRadient-based Algorithm for Non-Smooth Optimization',     ...
    'Version 1.6.4',                                                    ...
    'Licensed under the AGPLv3, Copyright (C) 2016-2020 Tim Mitchell'   };
end