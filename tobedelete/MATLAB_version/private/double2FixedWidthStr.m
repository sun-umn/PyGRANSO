function out = double2FixedWidthStr(varargin)  
%   double2FixedWidthStr:
%       Formats number x in the desired width field so that the length of
%       the string remains constant for all values of x and the most
%       significant digit is always in the second place (so sign changes do
%       not shift alignment and a '+' is not necessary).  The precision
%       displayed is always at least 2 digits and varies on the magnitude
%       of the number, as numbers requiring scientific notation require
%       more characters to display their magnitude.
%
%   USAGE:
%       1) Get a function handle to do formatting for a given width
%       format_fn   = double2FixedWidth(width);
%
%       2) Format a number to a string of a given width
%       str         = double2FixedWidth(number,width);
%
%   INPUT:
%       width       An integer between 9 and 23 specifying the number
%                   of chars to use for representing a number as a string.
%                   For width == 9, the string representation will always
%                   show the first two most significant digits while width
%                   == 23, the representation will show all the significant
%                   digits up to machine precision.
%
%       number      A double to convert to a fixed width string.
%
%   OUTPUT:         [Either one of the following] 
%       format_fn   Function handle that takes single argument x and
%                   converts it to a string representation with fixed width
%                   such that the most significant digit will always be the
%                   second character.  Note that if x is a number, variable
%                   precision is used to ensure the width of the string
%                   representation does not change.  Nonnumeric x will
%                   cause a centered 'n/a' to be returned.  Both infs 
%                   (positive and negative) and NaNs are supported.
%
%       str         fixed width aligned string representation of number.
%
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   double2FixedWidthStr.m introduced in GRANSO Version 1.0.
%
% =========================================================================
% |  double2FixedWidthStr.m                                               |
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

    % need places for
    % - sign (1 char)
    % - minimum of 2 digits with decimal point (3 chars)
    % - up to 5 chars for exponent e+111
    MIN_WIDTH   = 9;

    % must be able to display up to 16 digits
    % - sign (1 char)
    % - 16 digits + decimal point (17 chars)
    % - up to 5 chars for exponent
    MAX_WIDTH   = 23;
    
    if nargin == 1
        width   = varargin{1};
        assertWidth(width);
        out     = @(x) double2FixedWidth(x,width);
    elseif nargin == 2
        str     = varargin{1}; 
        width   = varargin{2};
        assertWidth(width);
        out     = double2FixedWidth(str,width);
    else
        error(  'double2FixedWidthStr:invalidInput',                    ...
                'double2FixedWidthStr: requires 1 or 2 input args.'     );
    end  
    
    function assertWidth(width)
        assert( width >= MIN_WIDTH && width <= MAX_WIDTH,               ...
                'double2FixedWidthStr:invalidInput',                    ...
                'double2FixedWidthStr: width must be in [9,10,...,23].' );
    end
end

function x_str = double2FixedWidth(x,width)
    if ~isnumeric(x)
        x_str       = sprintf(' n/a%s',repmat(' ',1,width-4));
        return
    elseif x == 0
        x_str       = sprintf(' 0.%s',repmat('0',1,width-3));
        return
    elseif x == inf
        x_str       = sprintf(' Inf%s',repmat(' ',1,width-4));
        return
    elseif x == -inf
        x_str       = sprintf('-Inf%s',repmat(' ',1,width-4));
        return
    elseif isnan(x)
        x_str       = sprintf(' NaN%s',repmat(' ',1,width-4));
        return
    end

    is_negative     = x < 0;
    x               = abs(x);
    n_whole         = nDigitsInWholePart(x);

    if x >= 1e+100 || x < 1e-99
        % this assumes x is positive
        x_str       = sprintf('%.*e', width - 8, x);
    elseif n_whole > width - 3 || x < 1e-3
        % this assumes x is positive
        x_str       = sprintf('%.*e', width - 7, x);
    else
        digits      = width - 2 - n_whole;
        if n_whole < 1
            digits  = digits - 1;
        end
    
        x_str       = sprintf('%.*f', digits, x);
        % get length without sign since the actual number of whole digits 
        % may be increased by one due to rounding in sprintf (e.g.
        % 99.99999999...)
        expected    = width - double(x >= 0);  
        if length(x_str) > expected
            x_str   = x_str(1:expected);
        end
    end
    
    if is_negative
        x_str       = ['-' x_str];
    else
        x_str       = [' ' x_str];
    end
end