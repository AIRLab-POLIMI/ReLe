function fprintfmat(A, every_x_column, every_x_row, format)
% Prints a matrix separating rows with '-' and columns with '|'.
%
% Inputs:
% - A                 : the matrix to be printed
% - every_x_column    : number of columns separated by '|'
% - every_x_row       : number of rows separated by '-'
% - format (optional) : the char format for the values of the matrix

if rem(size(A,2), every_x_column) ~= 0
    error('Incompatible number of columns')
end

if rem(size(A,1), every_x_row) ~= 0
    error('Incompatible number of rows')
end

if nargin == 3
    format = '%.3f';
end

column_block = [repmat([format ' '], 1, every_x_column) '| '];
str_row = repmat(column_block, 1, size(A, 2) / every_x_column);
str_row(end-2:end) = [];
row_delimiter = repmat('-', 1, length(sprintf(str_row, A(1,:))));
row_block = repmat([str_row '\n'], 1, every_x_row);
fprintf(['\n' row_delimiter '\n']);
fprintf([ row_block row_delimiter '\n' ], A);
fprintf('\n');
