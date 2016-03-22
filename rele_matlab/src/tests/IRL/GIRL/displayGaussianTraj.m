%% Print Trajectories
close all
clear

addpath(genpath('../../..'));

fraction = 1.0;
step = floor(1/fraction);

%% create gate
xs = 100;
ys = 120;

xe = 120;
ye = 100;

%% plot trajectories of imitator

%titles{1} = 'Imitator 1 -Normalization: None';
titles{1} = 'Imitator 2 -Normalization: Log Disparity';
lastindex = 1;


for i = 1:lastindex

    csv = csvread(['/tmp/ReLe/gaussian/GIRL/Imitator.log']);
    traj = readDataset(csv);

    figure(i)
    title(titles{i})
    xlabel('t')
    ylabel('x1')
    zlabel('x2')
    
    hold on;
    maxTime = 1;
    for episode = 1:step:size(traj,1)
        plot3(1:size(traj(episode).x,1), traj(episode).x(:, 1), traj(episode).x(:,2));
        maxTime = max(maxTime, size(traj(episode).x, 1));
    end
    
     %fill3([1 maxTime maxTime 1 1], [xs xs xe xe xs], [ys ys ye ye ys], 'y')

end

%% plot trajectories of expert
csv = csvread('/tmp/ReLe/gaussian/GIRL/Trajectories.log');
traj = readDataset(csv);

figure(lastindex + 1)
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

%fill3([1 700 700 1 1], [xs xs xe xe xs], [ys ys ye ye ys], 'y')