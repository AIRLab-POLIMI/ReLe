clc;
addpath ..
params.policyParameters = ones(6,1);
params.asVariance = 1;
params.nbRewards = 2;
params.penalize = 1;
params.initType = 'random_discrete';
tic;[samples, J, g, h] = collectSamples('dam', 1, 10, 1.0, params);toc;
% sum(samples(32).r,2)
% J(32,:)
[samples.s', samples.a', samples.nexts']

%%
clear params
params.dim = 2;
params.policyParameters = -0.5*ones(params.dim,1);
params.stddev = 1;
tic;[samples, J, g, h] = collectSamples('lqr', 1000, 150, 1.0, params);toc;