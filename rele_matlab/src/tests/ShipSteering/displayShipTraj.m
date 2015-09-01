%% ShipSteering data visualizer
%clear old data
clear

%clear old figures
close all

%% Choose file
basedir = '/tmp/ReLe/ship/PG/';
trajectoryFile = [basedir 'ship_r.log'];

%% create gate
xs = 100;
ys = 120;

xe = 120;
ye = 100;

%% plot trajectories of expert


csv = csvread(trajectoryFile);
traj = readDataset(csv);

figure(1)
title('Expert')
xlabel('t')
ylabel('x1')
zlabel('x2')

hold on;

maxTime = 1;
for episode = 1:size(traj,1)      
       plot3(1:size(traj(episode).x,1), traj(episode).x(:, 1), traj(episode).x(:,2));
       maxTime = max(maxTime, size(traj(episode).x, 1));
end

fill3([1 700 700 1 1], [xs xs xe xe xs], [ys ys ye ye ys], 'y')