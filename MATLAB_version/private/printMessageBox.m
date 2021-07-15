function printMessageBox(   use_ascii,  use_orange,     margin_spaces,  ...
                            title_top,  title_bottom,   msg_lines,      ...
                            sides,      user_width                      )
%   printMessageBox:
%       Prints a message to the console surrounded by a border.
%       If user_width is not provided, the border is automatically sized to
%       fit the maximum width line (including given whitespace formatting)
%       specified in the message.
%
%   INPUT:
%       use_ascii       logical: use standard ascii instead of extended
%                       characters to make the border
%
%       use_orange      logical: use orange text to print the border and 
%                       message
%
%       margin_spaces   nonnegative integer: number of spaces for left and
%                       right margins of the text
%
%       title_top       string or []: specify a title for the top border. 
%                       Empty or a blank string means no title.
%
%       title_bottom    string or []: specify a title for the top border. 
%                       Empty or a blank string means no title.
% 
%       msg_lines       cell array of each line to print in the message
%                       box.  White space formatting is observed.
%
%       sides           optional argument, logical: whether or not side 
%                       borders should be printed.  
%       
%       user_width      optional argument, positive integer such that:
%                           user_width must be >= 6 * 2*margin_spaces
%                       to specify the width of the box.  Note that the box
%                       width may increase beyond this value to accommodate
%                       top and bottom titles, if they would otherwise not
%                       completely fit.  The box's content however is
%                       treated differently.  If sides == true, lines in
%                       msg_lines that are too long will be truncated as
%                       necessary.  If sides == false, user_width is only
%                       used to determine the widths of the top/bottom
%                       borders and lines will be not truncated, even if
%                       they extend beyond the widths of the headers.
%             
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   printMessageBox.m introduced in GRANSO Version 1.0.
%
% =========================================================================
% |  printMessageBox.m                                                    |
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

    TITLE_INDENT        = 5;
    TITLE_MARGIN        = 1;
   
    if nargin < 7
        sides           = true;
    end
    have_user_width     = nargin > 7;
    if ~have_user_width
        user_width      = 0;
    end
    border_spaces       = 2*margin_spaces;
    fixed_width_headers = have_user_width;
    
    % process title and line widths
    max_line_width      = max(cellfun(@length,msg_lines));
    % strtrim does not accept [] so first append '' to all titles
    title_top           = strtrim([title_top '']);
    title_bottom        = strtrim([title_bottom '']);
    title_top_width     = length(title_top);
    title_bottom_width  = length(title_bottom);
    
    % min width of header, not including the ends (the corner chars)
    header_width        = max(title_top_width,title_bottom_width);
    if header_width > 0  
        header_width    = header_width + 2*(TITLE_INDENT + TITLE_MARGIN);
    end
    header_width        = max(header_width,user_width);
    if ~fixed_width_headers
        header_width    = max(header_width,max_line_width+border_spaces);
    end
    
    if ~have_user_width
        % increase max_line_width if the header width is longer
        max_line_width  = max(max_line_width,header_width-border_spaces);
    elseif sides
        % crop lines if they are too long for the user-specified box
        max_line_width  = header_width - border_spaces;
        trunc_fn        = @(s) truncateStr(s,max_line_width);
        msg_lines       = cellfun(  trunc_fn,           msg_lines,      ...
                                    'UniformOutput',    false           );
    end

    [top,bottom,vbar]   = getHeaders(use_ascii,header_width);
    % insert titles as necessary                                
    add_title_fn        = @(h,t) addTitle(h,t,TITLE_INDENT,TITLE_MARGIN);
    top                 = add_title_fn(top,title_top);
    bottom              = add_title_fn(bottom,title_bottom);    
    
    if ~sides
        vbar            = [];
    end
    format_str          = lineFormatStr(max_line_width,margin_spaces,vbar);
    
    if use_orange
        print_fn        = @printOrange;
    else
        print_fn        = @fprintf;
    end
    
    print_line_fn       = @(s) print_fn(format_str,s);
    
    print_fn(top);
    cellfun(print_line_fn,msg_lines);
    print_fn(bottom);
end

function s = truncateStr(s,width)
    if length(s) > width
        s = [s(1:width-3) '...'];
    end
end

function header = makeHeaderBar(width,hbar,lc,rc)
    header = [lc repmat(hbar,1,width) rc '\n'];
end

function header = addTitle(header,title,indent,spaces)
    if isempty(title)
        return
    end
    title_chars     = length(title) + 2*spaces;
    indx            = indent+2:indent+1+title_chars;
    header(indx)    = sprintf('%*s%s%*s',spaces,'',title,spaces,'');
end

function [top,bottom,vbar] = getHeaders(ascii,header_width)

    if ascii
        [hbar,vbar,lt,rt,lb,rb] = getSymbolsASCII();
    else
        [hbar,vbar,lt,rt,lb,rb] = getSymbolsExtended();
    end

    % set up top and bottom
    top             = makeHeaderBar(header_width,hbar,lt,rt);
    bottom          = makeHeaderBar(header_width,hbar,lb,rb);
end

function format_str = lineFormatStr(max_line_width,margin_spaces,vbar)
    margin          = repmat(' ',1,margin_spaces);
    if nargin < 3 || isempty(vbar)
        format_str  = [' ' margin '%s\n'];
    else
        vbar        = vbar(1);
        s_chars     = num2str(max_line_width);
        format_str  = [vbar margin  '%-' s_chars 's' margin vbar '\n'];
    end
end

function [hbar,vbar,lt,rt,lb,rb] = getSymbolsExtended()
    hbar    = char(hex2dec('2550'));
    vbar    = char(hex2dec('2551'));
    lt      = char(hex2dec('2554'));
    rt      = char(hex2dec('2557'));
    lb      = char(hex2dec('255A'));
    rb      = char(hex2dec('255D'));
end

function varargout = getSymbolsASCII()
    [varargout{1:6}] = deal('#');
end
