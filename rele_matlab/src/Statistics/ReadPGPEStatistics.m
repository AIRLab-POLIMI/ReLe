function stats = ReadPGPEStatistics(fp)
nbIterations = fscanf(fp, '%d', 1);
nbIterations = nbIterations - 1;
stats = repmat(struct('metaParams',[], 'metaGradient', [], ...
    'policies', []), nbIterations, 1);
for i = 1:nbIterations
    stats(i) = ReadPGPEIterationStats(fp);
end
end