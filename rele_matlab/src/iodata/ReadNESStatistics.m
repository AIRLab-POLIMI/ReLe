function [ iteration, index ] = ReadNESStatistics(csv, index)
%READNESSTATISTICS Read NES statistics
%   Read all the parameters used by the algorithm and their reward

[iteration, index] = ReadPGPEStatistics(csv, index);

nbDistParams = length(iteration.params);
iteration.fisher = csv(index:index+nbDistParams-1, 1:nbDistParams);

index = index + nbDistParams; %go to next episode

end