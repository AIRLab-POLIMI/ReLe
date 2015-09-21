%% Print Trajectories
close all
clear all

addpath(genpath('../../..'));

fraction = 1.0;
step = floor(1/fraction);

%% create gate
xs = 100;
ys = 120;

xe = 120;
ye = 100;

%% plot trajectories of imitator
csv = csvread('/tmp/ReLe/ship/EMIRL/TrajectoriesImitator.txt');
traj = readDataset(csv);

figure(1)
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


%% plot trajectories of expert
csv = csvread('/tmp/ReLe/ship/EMIRL/TrajectoriesExpert.txt');
traj = readDataset(csv);

figure(2)
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