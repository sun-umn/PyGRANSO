function lines = wrapToLines(str,width,indent)
%   wrapToLines:
%       Converts the string str into a cell array of lines, each of at most
%       length width by doing basic word wrapping (no hyphenation).  Each
%       line can be optionally indented.  Words will be truncated if their 
%       length exceeds the specified width minus the indent value.
%
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   wrapToLines.m introduced in GRANSO Version 1.0.
%
% =========================================================================
% |  wrapToLines.m                                                        |
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

    if ~ischar(str)
        error(  'wrapToLines:invalidInput',                         ...
                'wrapToLines: str must be a string of characters.'  );
    end
    if width < 4
        error(  'wrapToLines:invalidInput',                         ...
                'wrapToLines: width must be at least 4.'            );
    end    
    if indent < 0
        error(  'wrapToLines:invalidInput',                         ...
                'wrapToLines: indent must be nonnegative.'          );
    end
    if width - indent < 4 
        error(  'wrapToLines:invalidInput',                         ...
                'wrapToLines: width - indent must be at least 4.'   );
    end
    
    chars       = length(str);
    max_lines   = ceil(chars/width);

    words       = strToTrimmedWords(str,width);
    
    count       = 0;
    lines       = cell(max_lines,1);
    line        = getBlankLine(width);
    position    = 1 + indent;
    for j = 1:length(words)
        word        = words{j};
        chars       = length(word); 
        word_end    = position - 1 + chars;
        if word_end > width
            count           = count + 1;
            lines{count}    = line;
            line            = getBlankLine(width);
            position        = 1 + indent;
            word_end        = position - 1 + chars;
        end
        line(position:word_end) = word;
      
        if j == length(words)
            count           = count + 1;
            lines{count}    = line;
        elseif word(end) == '.' || word(end) == '?' || word(end) == '!'
            position = position + chars + 2;
        else
            position = position + chars + 1;
        end
    end
    lines = lines(1:count);
end

function line = getBlankLine(width)
    line = repmat(' ',1,width);
end

function w = trimWord(w,width)
    if length(w) < 5
        return
    end
    w               = w(1:width);
    w(end-2:end)    = '...';
end

function words = strToTrimmedWords(str,width)
    words                   = strsplit(str);   
    lengths                 = cellfun(@length,words);
    too_long_indx           = lengths > width;
    trim_word_fn            = @(w) trimWord(w,width);
    
    words(too_long_indx)    = cellfun(  trim_word_fn,                   ...
                                        words(too_long_indx),           ...
                                        'UniformOutput', false          );
  
end