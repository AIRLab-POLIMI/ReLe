%% Print Trajectories
close all
clear
clc

addpath(genpath('../../..'));

fraction = 1.0;
step = floor(1/fraction);

outPath = '/tmp/ReLe/matlab_out/ship/';
[~,~,~] = mkdir(outPath);


%% create gate
xs = 100;
ys = 120;

xe = 120;
ye = 100;

%% list algorithms
alg = {
    %'GIRL', ...
    %'ExpectedDeltaIRL',  ...
    'EGIRL', ...
    'EMIRL', ...
    'EpisodicExpectedDeltaIRL'};

lastindex = length(alg);


%% plot trajectories of imitator
for i = 1:lastindex
    figure(i)
    path = ['/tmp/ReLe/ship/', alg{i}, '/'];  
    
    %% plot trajectories of expert
    csv = csvread([path, 'TrajectoriesExpert.txt']);
    traj = readDataset(csv);
    
    subplot(2, 1, 1)
    title('Expert')
    xlabel('t')
    ylabel('x1')
    zlabel('x2')
    
    hold on;
    
    maxTime = 1;
    for episode = 1:step:size(traj,1)
        plot3(1:size(traj(episode).x,1), traj(episode).x(:, 1), traj(episode).x(:,2));
        maxTime = max(maxTime, size(traj(episode).x, 1));
    end
    
    fill3([1 700 700 1 1], [xs xs xe xe xs], [ys ys ye ye ys], 'y')  
    
    
    %% Plot imitator
    csv = csvread([path, 'TrajectoriesImitator.txt']);
    traj = readDataset(csv);
    
    subplot(2, 1, 2)
    title('Imitator')
    xlabel('t')
    ylabel('x1')
    zlabel('x2')
    
    hold on;
    maxTime = 1;
    for episode = 1:step:size(traj,1)
        plot3(1:size(traj(episode).x,1), traj(episode).x(:, 1), traj(episode).x(:,2));
        maxTime = max(maxTime, size(traj(episode).x, 1));
    end
    
    fill3([1 maxTime maxTime 1 1], [xs xs xe xe xs], [ys ys ye ye ys], 'y')
    
    %% set title and save figure
    suptitle(alg{i})
    savefig([outPath, alg{i},'.fig']);
end