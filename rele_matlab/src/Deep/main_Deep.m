%% Rocky data visualizer
addpath(genpath('../Statistics'));
addpath(genpath('./GradientTests'));

%clear old data
clear all;
clc;

nbEpisodes = 50;
nbUpdates  = 200;
stepLength = 0.001;
algorithm = 'gb';
domain = 'deep';

prog = ['/home/matteo/Projects/github/ReLe/rele-build/',domain,'_PG'];
args = [num2str(nbUpdates), ' ', num2str(nbEpisodes), ...
    ' ', num2str(stepLength)];

toExe = [prog ' ' algorithm ' ' args];
status = system(toExe);

%% Read data

disp('Reading data trajectories...')
csv = csvread(['/tmp/ReLe/',domain,'/PG/',domain,'_',algorithm,'.log']);

disp('Organizing data in episodes...')
episodes = readDataset(csv);
clearvars csv

%% Read agent data

clear data

disp('Reading agent data...')
csv = csvread(['/tmp/ReLe/',domain,'/PG/',domain,'_',algorithm,'_agentData.log']);

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
    
    %     if strcmp(algorithm, 'enac')
    %         policy.weights = data(i).params(1:end-1)';
    %     else
    policy.weights = data(i).params';
    policy.it = 1;
    gamma = 1;
    obj = 1;
    if strcmp(algorithm, 'r')
        dJdtheta = eREINFORCE(policy, episodes(start:start+step-1), gamma, obj, data(i));
    elseif strcmp(algorithm, 'rb')
        dJdtheta = eREINFORCEbase(policy, episodes(start:start+step-1), gamma, obj);
    elseif strcmp(algorithm, 'g')
        dJdtheta = gGPOMDP(policy, episodes(start:start+step-1), gamma, obj);
    elseif strcmp(algorithm, 'gb')
        dJdtheta = gGPOMDPbase(policy, episodes(start:start+step-1), gamma, obj);
    elseif strcmp(algorithm, 'gsb')
        dJdtheta = gGPOMDPsinglebase(policy, episodes(start:start+step-1), gamma, obj);
    elseif strcmp(algorithm, 'enac')
        dJdtheta = eNACbase1(policy, episodes(start:start+step-1), gamma, obj);
    end
    [data(i).gradient',dJdtheta]
    assert(norm(data(i).gradient'-dJdtheta,inf)<=1e-2);
    start = start + step;
end