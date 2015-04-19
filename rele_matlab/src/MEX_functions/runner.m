params.policyParameters = ones(6,1);
params.asVariance = 1;
params.nbRewards = 2;
params.penalize = 1;
params.initType = 'random_discrete';
tic;[samples, J, g, h] = collectSamples('dam', 1000, 150, 1.0, params);toc;
% sum(samples(32).r,2)
% J(32,:)
