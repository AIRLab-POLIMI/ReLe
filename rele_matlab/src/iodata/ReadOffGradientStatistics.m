function [ iteration, index ] = ReadOffGradientStatistics(csv, index)
%READOFFGRADIENTSTATISTICS Read off-policy gradient statistics
%   Read all the parameters used by the algorithm and their reward

[ iteration, index ] = ReadGradientStatistics(csv, index);

iteration.historyIW = csv(index,:);

index = index + 1; %go to next episode

end