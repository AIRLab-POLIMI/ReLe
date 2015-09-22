function [ q ] = computeDimensions( S, varMin )
%COMPUTEDIMENSIONS Summary of this function goes here
%   Detailed explanation goes here
for q = 1:size(S, 1)
    var = trace(S(1:q, 1:q))/trace(S);
    
    if(var > varMin)
       break;        
    end
end

end

