%% Rocky data visualizer
addpath(genpath('../Statistics'));
addpath(genpath('./GradientTests'));

%clear old data
clear all;

%% Read data

disp('Reading data trajectories...')
csv = csvread('/tmp/ReLe/Nls/PG/Nls.log');

disp('Organizing data in episodes...')
episodes = readDataset(csv);
clearvars csv

%% Read agent data

disp('Reading agent data...')
csv = csvread('/tmp/ReLe/Nls/PG/Nls_agentData.log');

disp('Organizing data...')

index = 1;
ep = 1;

while(index < size(csv, 1))
    [data(ep), index] = ReadGradientStatistics(csv, index);
    ep = ep + 1;
end

clearvars csv

data

%%
clc
start = 1;
step = length(episodes)/length(data);
for i = 1:length(data)
    policy.weights = data(i).params';
    policy.stddev = 0.2;
    gamma = 0.95;
    obj = 1;
%     dJdtheta = eREINFORCE(policy, episodes(start:start+step-1), gamma, obj, data(i));
%     dJdtheta = eREINFORCEbase(policy, episodes(start:start+step-1), gamma, obj);
%     dJdtheta = gGPOMDP(policy, episodes(start:start+step-1), gamma, obj);
    dJdtheta = gGPOMDPbase(policy, episodes(start:start+step-1), gamma, obj);
    [data(i).gradient',dJdtheta]
    assert(norm(data(i).gradient'-dJdtheta,inf)<=1e-2);
    start = start + step;
end