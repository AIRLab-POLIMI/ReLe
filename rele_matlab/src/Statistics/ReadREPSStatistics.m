function [ iteration, index ] = ReadREPSStatistics(csv, index)
%READREPSSTATISTICS Read REPS statistics
%   Read all the parameters used by the algorithm and their reward

index = index +1; %skip episode info
iteration.metaParams = csv(index, :);
index = index +1; 
individualsN = csv(index, 1);
    
for i=1:individualsN
    index = index +1;
    iteration.policies(i).policy = csv(index, :);
    index = index +1;
    iteration.policies(i).J = csv(index, 1);
end

index = index + 1; %go to next episode


end