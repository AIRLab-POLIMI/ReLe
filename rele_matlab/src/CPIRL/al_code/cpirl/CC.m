function [x, fval, result] = CC(A, b)
%% CHEBYSHEV CENTER of a polyhedron {x | Ax<=b}
%%

disp('#### CC ####');

%CHEBYCENTER Compute Chebyshev center of polytope Ax <= b.
%  The Chebyshev center of a polytope is the center of the largest
%  hypersphere enclosed by the polytope. 
%  Requires optimization toolbox.

[n,p] = size(A);
an    = sqrt(sum(A.^2,2));

A1        = zeros(n+1,p+1);
A1(1:n,1:p) = A;
A1(1:n,p+1) = an;
A1(n+1,p+1) = -1;

b1 = [b;0];

f      = zeros(p+1,1);
f(p+1) = -1;

tic;
[x, fval, exitflag] = cplexlp(f, A1, b1);
t_cc = toc;

if nargout > 2
    result.radius  = -fval;
    result.op_time = t_cc;
    if (exitflag < 0)
        result.exitflag = -1;
    else
        result.exitflag = 0;
    end
end

x = double(x(1:p));

disp('#### Ended ####');

return