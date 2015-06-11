function [ iteration, index, stack ] = ReadHierarchicalGradientStatistics(csv, index)
%READHIERARCHICALGRADIENTSTATISTICS Summary of this function goes here
%   Detailed explanation goes here

index = index +1; %skip episode info
nbparameters = csv(index, 1);
index = index +1;
nbepisode = csv(index, 1);
index = index +1;

%read stack information
stackSize = csv(index, 1);
index = index +1;

stack = cell(stackSize, 1);

for i=1:stackSize
    stackDepth = csv(index, 1);
    stack{i} = csv(index, 2:stackDepth);
    index = index + 1;
end


%Read J and gradients
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

