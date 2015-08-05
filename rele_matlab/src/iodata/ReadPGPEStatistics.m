function [ iteration, index ] = ReadPGPEStatistics(csv, index)
%READPGPESTATISTICS Read PGPE statistics
%   Read all the parameters used by the algorithm and their reward

index = index +1; %skip episode info
nbparameters = csv(index, 1);
index = index +1;
iteration.params = csv(index, 1:nbparameters);
index = index +1;
nbPolParams = csv(index, 1);
index = index +1;
nbEpisodesPerPolicy = csv(index, 1);
index = index +1;
individualsN = csv(index, 1);

for i=1:individualsN
    index = index +1;
    iteration.policies(i).policy = csv(index, 1:nbPolParams);
    index = index +1;
    iteration.policies(i).J = csv(index, 1:nbEpisodesPerPolicy);  
    index = index +1;
    iteration.policies(i).diffdist = csv(index, 1:nbparameters); 
end

index = index + 1;
iteration.gradient = csv(index, 1:nbparameters);
index = index +1;
nbElStepLength = csv(index, 1);
index = index + 1;
iteration.stepLength = csv(index, 1:nbElStepLength);

index = index + 1; %go to next episode

end