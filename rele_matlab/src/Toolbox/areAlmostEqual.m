% Checks if two matrices are almost equal.
%
% Input: 
%         - A         : N-by-N matrix
%         - B         : N-by-N matrix
%         - tolerance : a threshold of tolerance
%
% Output: given C = |A - B|, the function returns 1 if c(i,j) < tolerance
%         for all (i,j)
function b = areAlmostEqual( A, B, tolerance )

b = max(max(abs(A-B))) < tolerance;

end
