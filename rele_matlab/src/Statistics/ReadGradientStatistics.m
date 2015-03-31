function [ iteration, index ] = ReadGradientStatistics(csv, index)
%READGRADIENTSTATISTICS Read gradient statistics
%   Read all the parameters used by the algorithm and their reward

index = index +1; %skip episode info
nbparameters = csv(index, 1);
index = index +1;
nbepisode = csv(index, 1);
index = index +1;
iteration.J = csv(index, 1:nbepisode);

for i=1:nbepisode
    index = index +1;
    iteration.histGradient(i).g = csv(index, 1:nbparameters);
end

index = index + 1;
iteration.params = csv(index, 1:nbparameters);
index = index + 1;
iteration.gradient = csv(index, 1:nbparameters);
index = index + 1;
iteration.stepLength = csv(index, 1);

index = index + 1; %go to next episode

end