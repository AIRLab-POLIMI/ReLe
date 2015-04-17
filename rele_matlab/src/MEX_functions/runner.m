params.policyParameters = ones(5,1);
params.asVariance = 1;
params.nbRewards = 2;
params.penalize = 1;
params.initType = 'random_discrete';
tic;[samples, J] = collectSamples('dam', 1000, 150, 1.0, params);toc;
