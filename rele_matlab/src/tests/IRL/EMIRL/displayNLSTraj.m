%% Print Trajectories
close all
clear all

addpath(genpath('../../..'));

fraction = 0.1;
step = floor(1/fraction);

%% create cylinder
[yc, zc, xc] = cylinder();

xc = 80*xc;
yc = 0.1*yc;
zc = 0.1*zc;

%% plot trajectories of imitator

lastindex = 2;

csv = csvread('/tmp/ReLe/nls/EMIRL/TrajectoriesImitator.txt');
traj = readDataset(csv);

figure(1)
title('Imitator')
xlabel('t')
ylabel('x1')
zlabel('x2')

hold on;
mesh(xc, yc, zc, 'FaceColor','none', 'EdgeColor','red')

for episode = 1:step:size(traj,1)
    
    if(size(traj(episode).x, 2) == 1)
        plot(traj(episode).x);
    else if(size(traj(episode).x, 2) == 2)
            plot3(1:size(traj(episode).x,1), traj(episode).x(:, 1), traj(episode).x(:,2));
        end
    end
end

%% plot trajectories of expert
csv = csvread('/tmp/ReLe/nls/EMIRL/TrajectoriesExpert.txt');
traj = readDataset(csv);

figure(2)
title('Expert')
xlabel('t')
ylabel('x1')
zlabel('x2')

hold on;
mesh(xc, yc, zc, 'FaceColor','none', 'EdgeColor','red')

for episode = 1:step:size(traj,1)
    
    if(size(traj(episode).x, 2) == 1)
        plot(traj(episode).x);
    else if(size(traj(episode).x, 2) == 2)            
            plot3(1:size(traj(episode).x,1), traj(episode).x(:, 1), traj(episode).x(:,2));
        end
    end
end