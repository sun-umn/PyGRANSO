function printer = tablePrinter(    use_ascii,  use_orange,             ...
                                    labels,     widths,                 ...
                                    spacing,    span_labels             )
                                
%   tablePrinter:
%       An object that makes it much easier to print nice looking tables to
%       MATLAB's console.  
%
%   INPUT:
%       use_ascii       
%           Logical to indicate whether or not to use basic ascii
%           characters instead of the extended characters to print table
%           boundaries.
% 
%       use_orange
%           Logical to indicate whether or not to enable the orange 
%           printing features; orange printing is an undocumented feature
%           in MATLAB hence the option to disable it if necessary.
%           As a shortcut, setting use_orange = [] is equivalent to false.
%
%       labels
%           Cell array of strings, to label each column in the table.  New
%           lines are supported in these labels.
%
%       widths
%           Array of positive integers, with each entry indicating the 
%           printable width (in number of characters) of each column in the 
%           table.  Labels will be truncated as necessary if their widths
%           exceed their corresponding column widths specified here. 
% 
%       spacing
%           A nonnegative integer indicating the number of spaces to use as
%           left/right border margins for all entries.
%
%       span_labels (optional)
%           A cell array of cell arrays to specify labels that should span
%           multiple columns.  Each entry must have the form:
%               {span_label_str,column_start,column_end}
%           Thus {'Group 1 data',2,4} would put span label 'Group 1 data' 
%           centered above the regular labels for columns 2 through 4. 
%           Overlapping span labels are NOT supported.
%   
%   OUTPUT: 
%       printer 
%           A tablePrinter object with the following methods:
%
%           .header()
%               Prints a table header of all the column labels.  This can
%               be called multiple times, such as every twenty rows.
%           
%           .row(col1_data,...,colk_data)
%               Prints a row in the table.  The number of provided input 
%               arguments must match the number of columns in the table. 
%               Each argument should be a string that represents the data 
%               to be printed for each column.  Each string must have a 
%               printable length that matches the width specified for its
%               given column (given by the input argument widths); if this 
%               not done, or new lines appear in these strings, then the
%               alignment of the table will be broken.  Each input string
%               is processed as fprintf %s argument.
% 
%           .msg(msg)
%               Prints a message inside the table, using its full width.
%               This function takes either a string or cell array, the
%               latter where each entry specifies a line in a multi-line
%               message.  Any line too long to be printed in the table will
%               be truncated.
%
%           .msgOrange(msg)
%               Same as .msg but will print the message using orange text.
% 
%           .overlay(msg)
%               Prints a message overlaid on the table.  Though this
%               produces a different look of the message, this function 
%               takes the same input arguments as .msg.
%
%           .overlayOrange(msg)
%               Same as .overlay but prints the overlay message using
%               orange text.
%       
%           .msgWidth()
%               Get the number of printable characters for a single line
%               in a message printed using .msg.
%
%           .overlayWidth()
%               Get the number of printable characters for a single line
%               in a message printed using .overlay.  Note that overlay
%               messages have less available width than regular messages.
% 
%           .close()
%               Close the current table, that is, print the bottom border.
%               Note that there is no need to explictly open/begin a table;
%               this is handled automatically.  Merely requesting something
%               is a printed (a header, a row, or a message) will print
%               also print the top border of a table as necessary, such as
%               if no printing has already been done or if .close() was
%               called to close a preceding table.
%       
%
%   For comments/bug reports, please visit the GRANSO GitLab webpage:
%   https://gitlab.com/timmitchell/GRANSO
%
%   tablePrinter.m introduced in GRANSO Version 1.0.
%
% =========================================================================
% |  tablePrinter.m                                                       |
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
                                                         
    if nargin < 6
        span_labels = [];
    end
    
    if isempty(use_orange)
        use_orange  = false;
    end
    
    [header,rules,row_str,msg_str,msg_width] = init(    use_ascii,      ...
                                                        labels,         ...
                                                        widths,         ...
                                                        spacing,        ...
                                                        span_labels     ); 
                                                    
    [printOverlayFn,overlay_width]  = overlayPrinter(rules.m.rs,msg_width);
                                                  
    last_printed = 'n';
  
    printer = struct(                                                   ...
                'header',           @printHeader,                       ...
                'row',              @printRow,                          ...
                'msg',              @(s) printMessage(false,s),         ...
                'msgOrange',        @(s) printMessage(use_orange,s),    ...
                'overlay',          @(s) printOverlay(false,s),         ...
                'overlayOrange',    @(s) printOverlay(use_orange,s),    ...
                'msgWidth',         @() msg_width,                      ...
                'overlayWidth',     @() overlay_width,                  ...
                'close',            @printClose                         );
                             
    function printHeader()
        switch last_printed
            case 'n'
                r = rules.t.h;
            case 'm'
                r = rules.m.fh;
            case {'r','h','w'}
                r = rules.m.rh;
        end
        fprintf(r);
        fprintf(header);
        
        last_printed = 'h';
    end

    function printRow(varargin)
        switch last_printed
            case 'n'
                r = rules.t.r;
            case 'm'
                r = rules.m.fr;
            case {'h','w'}
                r = rules.m.r;
        end
        if last_printed ~= 'r'
            fprintf(r);
        end
        
        fprintf(row_str,varargin{:});
        last_printed = 'r';
    end

    function printMessage(use_orange,msg_lines)
        if length(msg_lines) < 1
            return
        end
        switch last_printed
            case 'n'
                r = rules.t.f;
            case 'm'
                r = rules.m.f;
            case {'r','h','w'}
                r = rules.m.rf;
        end
        fprintf(r);
        if ischar(msg_lines)
            msg_lines   = {msg_lines};
        end
        if use_orange 
            msg_line_fn = @messageLineOrange;
        else
            msg_line_fn = @messageLineBlack;
        end
        cellfun(msg_line_fn, msg_lines);
        last_printed    = 'm';
    end

    function printOverlay(use_orange,msg_lines)
        if length(msg_lines) < 1
            return
        end
        switch last_printed
            case 'n'
                r = rules.t.r;
            case 'm'
                r = rules.m.fr;
            case {'r','h','w'}
                r = rules.m.r;
        end
        fprintf(r);
        printOverlayFn(use_orange,msg_lines);
        last_printed    = 'w';
    end

    function printClose()
        switch last_printed
            case 'm'
                r = rules.b.f;
            case {'r','h','w'}
                r = rules.b.r;
        end
        if last_printed ~= 'n'
            fprintf(r);
            last_printed = 'n';
        end
    end

    function messageLineBlack(str)
        str = prepareMessageStr(msg_width,msg_width,str);
        fprintf(msg_str,str);
    end

    function messageLineOrange(str)
        str = prepareMessageStr(msg_width,msg_width,str);
        if use_orange && numel(str) > 0
            str = formatOrange(str);
        end
        fprintf(msg_str,str);
    end  
end

function [print_overlay_fn, overlay_width] = overlayPrinter(rule,msg_width)
    % msg_width minus the first 3 chars of the rule, and then 2*3 chars
    % for 3 space margins on each side of a message line
    overlay_width       = msg_width - 9;
    print_overlay_fn    = @printOverlay;

    function printOverlay(use_orange,msg_lines)
        if ischar(msg_lines)
            msg_lines   = {msg_lines};
        end
        max_width       = max(cellfun(@length,msg_lines));
        cellfun(@printOverlayLine, msg_lines); 
           
        function printOverlayLine(str)
            str = prepareMessageStr(overlay_width,max_width,str);
            if use_orange && numel(str) > 0
                str = formatOrange(str);
            end
            fprintf([rule(1:3) '   ' str '   ' rule(max_width + 10:end)]);
        end
    end
end

function str = prepareMessageStr(print_width,max_width,str)
    chars   = length(str);
    if print_width < chars
        str = truncateStr(str,print_width);
    else % the string is not long enough so pad right side
        str = sprintf('%-*s',max_width,str);
    end
end

function [processed,n] = processLabel(label,width,add_arrows)
    label           = sprintf(label);
    splitted        = strsplit(label,'\n');
    processed       = cellfun(  @(x) truncate(x,width), splitted,   ...
                                'UniformOutput',        false       );
    if add_arrows && ~isempty(strtrim(label)) 
        process_fn  = @(x) addArrows(x,width);
    else
        process_fn  = @(x) centerString(x,width);
    end
    processed       = cellfun(  process_fn,         processed,      ...
                                'UniformOutput',    false           );
    n               = length(processed);    
end

function c = getEmptyStrings(n)
    c = cell(n,1);
    if n > 0 
        c = cellfun(@(x) '', c,'UniformOutput',false);
    end
end

function spannedLabel = spanLabel(label,lines)
    spannedLabel    = [getEmptyStrings(lines - length(label)); label(:)];
end

function label_array = processLabels(labels,widths,add_arrows)
    [labels,lines]  = cellfun(  @(l,w) processLabel(l,w,add_arrows),    ...
                                labels, num2cell(widths),               ...
                                'UniformOutput',false                   );
    labels          = cellfun(  @(x) spanLabel(x,max(cell2mat(lines))), ...
                                labels,                                 ...
                                'UniformOutput',false                   );
    label_array     = horzcat(labels{:});
end

function w = getTotalWidth(widths,delim_width,col_start,col_end)
    w = sum(widths(col_start:col_end)) + delim_width*(col_end-col_start);
end

function [labels,spans,col_starts] = parseSpanningLabels(spanning_labels)
    [labels,spans,col_starts] = cellfun(@parseSpanningLabel,        ...
                                        spanning_labels,            ...
                                        'UniformOutput',false       );
end

function [label,span,start] = parseSpanningLabel(spanning_label)
    label   = spanning_label{1};
    start   = spanning_label{2};
    span    = [start spanning_label{3}];
end

function labels = moveSpanLabels(span_labels,indx,n_cols)
    [labels{1:n_cols}]  = deal('');      
    labels(indx)        = span_labels;
end

function [labels,row_str] = processSpannedLabels(   labels,             ...
                                                    widths,             ...
                                                    span_labels,        ...
                                                    row_str,            ...
                                                    vs                  )
                                                            
    [span_labels,spans] = parseSpanningLabels(span_labels);
    
    delim_width         = length(vs);
    [n_lines,n_cols]    = size(labels);
    n_multicols         = length(spans);
    
    del_indx            = false(1,n_cols);
    span_indx           = false(1,n_cols);
    
    for j = 1:n_multicols
        span        = spans{j};
        col_start   = span(1);
        col_end     = span(2);
        span        = col_start:col_end;
           
        % join column labels for multicolumns and always do joins for the
        % the very last line
        for k = 1:n_lines
            labels_to_join          = labels(k,span);
            if k == n_lines || ~allEmpty(labels_to_join)
                labels{k,col_start} = strjoin(labels_to_join,vs);
            end
        end
        span_indx(col_start)    = true;
        widths(col_start)       = getTotalWidth(    widths,             ...
                                                    delim_width,        ...
                                                    col_start,          ...
                                                    col_end             );
        row_str{col_start}      = strjoin(row_str(span),vs);     
        del_indx(span(2:end))   = true;
    end
    
    % delete columns that have been joined into first column of each span
    labels(:,del_indx)  = [];
    widths(del_indx)    = [];
    row_str(del_indx)   = [];
    span_indx(del_indx) = [];
    
    % reset span_labels positions and format them
    n_cols              = length(widths);
    span_labels         = moveSpanLabels(span_labels,span_indx,n_cols);
    span_labels         = processLabels(span_labels,widths,true);
    
    % merge both sets of labels together
    overlap_line        = findOverlap(labels,span_indx);
    labels              = mergeLabels(span_labels,labels,overlap_line); 
end

function tf = allEmpty(cell_of_strs)
    s = strtrim(strjoin(cell_of_strs));
    tf = isempty(s);
end

function indx = getLabelLocations(labels)
    indx = cellfun(@(x) any(~isspace(x)),labels);
end

function line = findOverlap(labels,spanned_indx)
    lines   = size(labels,1);
    line    = 0;
    % don't allow the last line to overlap 
    for j = 1:lines-1
        line_labels     = labels(j,spanned_indx);
        nonempty_indx   = cellfun(@(x) any(~isspace(x)),line_labels);
        if any(nonempty_indx)
            break
        end
        line = j;
    end 
end

function labels = mergeLabels(labels_top,labels_bottom,overlapping_lines)
    [n_top,n_cols]  = size(labels_top);
    n_bottom        = size(labels_bottom,1);
    n_lines         = n_top + n_bottom - overlapping_lines;
    
    labels          = cell(n_lines,n_cols);
    [labels{:}]     = deal('');
   
    indx_bottom     = getLabelLocations(labels_bottom);
    indx            = [ false(n_lines-n_bottom,n_cols); indx_bottom ];
    labels(indx)    = labels_bottom(indx_bottom);
    
    indx_top        = getLabelLocations(labels_top);
    indx            = [ indx_top; false(n_lines-n_top,n_cols) ];
    labels(indx)    = labels_top(indx_top);
   
end

function [vs,vd] = getSymbolsVertical(use_ascii)
    if use_ascii
        vs  = '|';
        vd  = '|';
    else
        vs  = char(hex2dec('2502'));
        vd  = char(hex2dec('2551'));
    end
end

function [vs,vd,width] = getDelimitersVertical(use_ascii,spacing)
    [vs,vd] = getSymbolsVertical(use_ascii);
    space   = blanks(spacing);
    vs      = [space vs space];
    vd      = [space vd space];
    width   = length(vd);
end

function [header,rules,row_str,msg_str,msg_width] = init(   use_ascii,  ...
                                                            labels,     ...
                                                            widths,     ...
                                                            spacing,    ...
                                                            span_labels )

    [vs,vd]                 = getDelimitersVertical(use_ascii,spacing);
    
    n_cols                  = length(labels);
    labels                  = processLabels(labels,widths,false);
    [row_str{1:n_cols}]     = deal('%s');
    
    if ~isempty(span_labels)
        [labels,row_str]    = processSpannedLabels( labels,             ...
                                                    widths,             ...
                                                    span_labels,        ...
                                                    row_str,            ...
                                                    vs                  );
    end
    
    indx            = getLabelLocations(labels);
    widths          = cellfun(  @length, labels(end,:));
    blank_strs      = arrayfun( @blanks, widths, 'UniformOutput', false);
    blank_strs      = repmat(blank_strs,size(labels,1));
    labels(~indx)   = blank_strs(~indx); 
    
    % transpose output to a row cell array since mat2cell outputs to a
    % column and strjoin on 2014a will only accept rows, while on 2015a,
    % strjoin can handle either.
    lines           = mat2cell( labels,ones(1,size(labels,1)) ).';
   
    header_strs     = cellfun(  @(x) formatLine(x,vd),      ...
                                lines,                      ...
                                'UniformOutput',false       );
    header_last     = deblank(sprintf(header_strs{end,1}));
    header          = strjoin(header_strs,'');                    
    row_str         = formatLine(row_str,vd);
     
    rules           = makeRules(use_ascii,header_last);
    msg_width       = length(header_last) - 2;
    msg_str         = ['%-' num2str(msg_width) 's' vd '\n'];  
end

function label = addArrows(label,width)
    label       = strtrim(label);
    freespaces  = width - length(label);
    if freespaces < 0
        label   = label(1:width);
    elseif freespaces > 5
        arrow_length_left       = (freespaces - 4)/2;
        arrow_length_right      = arrow_length_left;
        if mod(freespaces,2) > 0 
            arrow_length_left   = ceil(arrow_length_left);
            arrow_length_right  = floor(arrow_length_right);
        end
        label = [   '<' repmat('-',1,arrow_length_left) ' '     ...
                    label                                       ...
                    ' ' repmat('-',1,arrow_length_right) '>'    ];
    else
        label = centerString(label,width);
    end
end

function line = formatLine(formatted_labels,vd)
    line = strjoin([formatted_labels '\n'],vd);
end

function rules = makeRules(use_ascii,h2)

    [vs,vd]             = getSymbolsVertical(use_ascii);
    get_symbols_fn      = ternOp(use_ascii,@getSymbolsASCII,@getSymbols);
    [   hs, hd,                         ...
        cs, cd, csd, cds,               ...
        tsdl, tdl, tds, td, tdsu, tdu,  ...
        edtr, edbr                      ] = get_symbols_fn();
    
    vd_indx             = strfind(h2,vd);
    vd_indx             = vd_indx(1:end-1);
    vs_indx             = strfind(h2,vs);
    
    flat_m1             = repmat(hd,1,length(h2)-1);

    % top rules, with downward corner piece at the end
    top_f               = [flat_m1 edtr '\n'];  % top rule - flat 
    
    top_h               = top_f;                % top rule - header
    top_h(vd_indx)      = td;
    
    top_r               = top_h;                % top rule - row
    top_r(vs_indx)      = tds;
    
    % bottom rules, with upward corner piece at the end
    bottom_f            = [flat_m1 edbr '\n'];  % bottom rule - flat
    
    bottom_r            = bottom_f;
    bottom_r(vd_indx)   = tdu;                  % bottom rule - header/row
    bottom_r(vs_indx)   = tdsu;
    
    % mid rules, with left-pointing T piece at the end 
    mid_f               = [flat_m1 tdl '\n'];   % mid rule - flat to flat
    
    mid_fh              = mid_f;                % mid rule - flat to header
    mid_fh(vd_indx)     = td;
    
    mid_fr              = mid_fh;               % mid rule - flat to row
    mid_fr(vs_indx)     = tds;
    
    mid_r               = mid_f;                % mid rule - row to row
    mid_r(vd_indx)      = cd;                   % (also header to row)
    mid_r(vs_indx)      = cds;
    
    mid_rs              = strrep(mid_r,hd,hs);  % mid rule - row to row
    mid_rs(vd_indx)     = csd;                  % single line, not double
    mid_rs(vs_indx)     = cs;
    mid_rs              = strrep(mid_rs,tdl,tsdl);
    
    mid_rh              = mid_f;                % mid rule - row to header
    mid_rh(vd_indx)     = cd;
    mid_rh(vs_indx)     = tdsu;
    
    mid_rf              = mid_f;                % mid rule - row to flat
    mid_rf(vd_indx)     = tdu;                  % (also header to flat)
    mid_rf(vs_indx)     = tdsu;
    
    t = struct( 'f',    top_f,      'h',    top_h,      'r',    top_r);
    b = struct( 'f',    bottom_f,   'r',    bottom_r);
    m = struct( 'f',    mid_f,      'fh',   mid_fh,     'fr',   mid_fr, ...
                'r',    mid_r,      'rh',   mid_rh,     'rf',   mid_rf, ...
                'rs',   mid_rs                                          );
            
    rules = struct('t',t,'b',b,'m',m);        
 
end

function [  hs, hd,                         ...
            cs, cd, csd, cds,               ...
            tsdl, tdl, tds, td, tdsu, tdu,  ...
            edtr, edbr                      ] = getSymbols()
        
    hs                  = char(hex2dec('2500'));
    hd                  = char(hex2dec('2550'));

    cs                  = char(hex2dec('253C'));
    cd                  = char(hex2dec('256C'));
    csd                 = char(hex2dec('256B'));
    cds                 = char(hex2dec('256A'));

    tsdl                = char(hex2dec('2562'));
    tdl                 = char(hex2dec('2563'));
    tds                 = char(hex2dec('2564'));
    td                  = char(hex2dec('2566'));
    tdsu                = char(hex2dec('2567'));
    tdu                 = char(hex2dec('2569'));

    edtr                = char(hex2dec('2557'));
    edbr                = char(hex2dec('255D'));

end

function [  hs, hd,                         ...
            cs, cd, csd, cds,               ...
            tsdl, tdl, tds, td, tdsu, tdu,  ...
            edtr, edbr                      ] = getSymbolsASCII()
        
    hs                  = '-';
    hd                  = '=';

    cs                  = '|';
    cd                  = '|';
    csd                 = '|';
    cds                 = '|';

    tsdl                = '-';
    tdl                 = '=';
    tds                 = '=';
    td                  = '=';
    tdsu                = '=';
    tdu                 = '=';

    edtr                = '=';
    edbr                = '=';

end