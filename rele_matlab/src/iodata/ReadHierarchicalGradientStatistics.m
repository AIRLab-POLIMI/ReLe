function [ iteration, index] = ReadHierarchicalGradientStatistics(csv, index)
%READHIERARCHICALGRADIENTSTATISTICS Summary of this function goes here
%   Detailed explanation goes here

index = index +1; %skip episode info
nbparameters = csv.readLine(index, 1);
index = index +1;
nbepisode = csv.readLine(index, 1);
index = index +1;

%Read J and gradients
iteration.J = csv.readLine(index, 1, nbepisode);

for i=1:nbepisode
    index = index +1;
    iteration.histGradient(i).g = csv.readLine(index, 1, nbparameters);
end

index = index + 1;
iteration.params = csv.readLine(index, 1, nbparameters);
index = index + 1;
iteration.gradient = csv.readLine(index, 1, nbparameters);
index = index + 1;
iteration.stepLength = csv.readLine(index, 1);
index = index + 1;

%read stack information
stackSize = csv.readLine(index, 1);
index = index +1;

stack = cell(stackSize, 1);

for i=1:stackSize
    stackDepth = csv.readLine(index, 1);
    stack{i} = csv.readLine(index, 2, stackDepth);
    index = index + 1;
end

end

