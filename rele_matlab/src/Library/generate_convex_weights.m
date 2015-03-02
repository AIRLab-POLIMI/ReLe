function W = generate_convex_weights ( N, step )
% GENERATE_CONVEX_WEIGHTS 
%
% INPUTS:
%         - N    : the number of elements to be weighted.
%         - step : the inverse of the stepsize in the weights interval.
%
% OUTPUT: 
%         - W    : a M-by-N matrix which rows are the weights for convex 
%                  combinations between N elements. M depends on 'step'.

W = recursiveLoops([], zeros(1,N), 1, N, step);

end

% Uses recursion to generate N-1 nested loops.
function W = recursiveLoops ( W, w, n, N, step )

if n == N
    v = 0;
    for i = 1 : n - 1
        v = v + w(i);
    end
    w(n) = 1 - v;
    W = [W; w];
    return
end

v = 0;
for i = 1 : n - 1
    v = v + w(i);
end
v = 1 - v;

for i = 0 : 1 / step : v
    w(n) = i;
    W = recursiveLoops(W,w,n+1,N,step);
end

end