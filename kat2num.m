function [X, dicti_cell, label_count] = kat2num(X, column, label_count) 

% Converts the char values of an cell array X(:,column) into doubles. 
% The function can convert as many different char categories as the
% label_count variable indicates. 
% The function outputs the new cell array X and the cell array dicti_cell 
% in which one can find the old category names and its corresponding 
% conversions.
%
% (C) Oliver Frank, 2021
% Technical University Berlin 
% 
% Free to Use!


    % Input Control 
    if ~exist('X', 'var') || ~iscell(X)
        error('The function needs a cell array to operate on')
    end
    
    if ~exist('column', 'var')
        error('Function needs the column number of the cell array')
    end
    
    if ~exist('label_count', 'var')
        label_count = 2;
    end
    
    
    % Preallocate the dicitionary cell for the conversions
    dicti_cell = cell(label_count,2);
    % Do the conversion as often as the label_count variable indicates
    for i = 1:label_count
        % count through all rows of the array and look for a char 
        % as soon as one is found break it
        for j = 1:height(X)        
            if ischar(X{j,column})
                label_name = X(j,column);
                X{j,column} = i;
                break 
            end      
        end 
        % use the so found char to convert all equal cells in the defined
        % array into the same number as a placeholder 
        for k = 1:height(X)      
            if strcmp(label_name,X(k,column))
                X{k,column} = i;
            end        
        end 
        % Put the value of the converted char and its corresponding number
        % into a cell to understand its meaning later 
        dicti_cell{i,1} = label_name;
        dicti_cell{i,2} = i;
        % Give user a variable to display for knowing the conversions  
        % to_know = ['The char ', dicti_cell{i,1}, ...
        %     ' corresponds to the double ', num2str(dicti_cell{i,2})];
        % disp(to_know)
    end 
    
end
        
        
        
    