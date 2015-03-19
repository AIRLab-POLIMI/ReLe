function iteration = ReadNESIterationStats(fp)
% Read metaparms
step = fscanf(fp, '%d', 1);
term = fscanf(fp, '%d', 1);
nbmetaparams = fscanf(fp, '%d', 1);
if feof(fp)
    iteration = [];
    return;
end
metaParams = fscanf(fp, '%f', nbmetaparams);
iteration.metaParams = metaParams;
metaGradient = fscanf(fp, '%f', nbmetaparams);
iteration.metaGradient = metaGradient;

% Read individuals
nbpolindividuals = fscanf(fp, '%d', 1);
for i = 1:nbpolindividuals
    iteration.policies(i) = ReadPGPEPolicyIndividual(fp);
end
iteration.F = zeros(nbmetaparams,nbmetaparams);
for i = 1:nbmetaparams
    for j = 1:nbmetaparams
        iteration.F(i,j) = fscanf(fp, '%f', 1);
    end
end
end