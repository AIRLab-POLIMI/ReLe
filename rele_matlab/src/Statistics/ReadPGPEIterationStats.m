function iteration = ReadPGPEIterationStats(fp)
% Read metaparms
step = fscanf(fp, '%d', 1);
term = fscanf(fp, '%d', 1);
nbmetaparams = fscanf(fp, '%d', 1);
metaParams = fscanf(fp, '%f', nbmetaparams);
iteration.metaParams = metaParams;
metaGradient = fscanf(fp, '%f', nbmetaparams);
iteration.metaGradient = metaGradient;

% Read individuals
nbpolindividuals = fscanf(fp, '%d', 1);
for i = 1:nbpolindividuals
    iteration.policies(i) = ReadPGPEPolicyIndividual(fp);
end
end