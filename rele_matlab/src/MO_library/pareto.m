% Filter a set of policies P and corresponding solutions S according to the
% Pareto dominance. Works only with 2 and 3 dimensions.
function [ s, p ] = pareto( s, p )

if nargin == 1
    p = [];
end

for i = size(s,1) : -1 : 1
    for j = size(s,1) : -1 : 1
        if size(s,2) == 2
            if s(j,1) >= s(i,1) && s(j,2) >= s(i,2) && i ~= j
                s = removerows(s,i);
                if nargin == 2
                    p(i) = [];
                end
                break
            end
        else
            if s(j,1) >= s(i,1) && s(j,2) >= s(i,2) && s(j,3) >= s(i,3) && i ~= j
                s = removerows(s,i);
                if nargin == 2
                    p(i) = [];
                end
                break
            end
        end
    end
end

end