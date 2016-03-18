%% Print Trajectories
close all
clear all

addpath(genpath('../../..'));

fraction = 1;
step = floor(1/fraction);

%% plot trajectories of expert
%algorithm = 'EMIRL'
algorithm = 'EGIRL';

path = ['/tmp/ReLe/lqr/', algorithm, '/'];


csv = csvread([path, '/TrajectoriesExpert.txt']);
traj = readDataset(csv);

figure(2)
title('Expert')
xlabel('x1')
ylabel('x2')
zlabel('x3')

hold on;

for episode = 1:step:size(traj,1)        
    plot3(traj(episode).x(:, 1), traj(episode).x(:,2), traj(episode).x(:, 3));
end

axis([0, 20, 0, 20])
zlim([0 20])

%% plot Parameters of expert
theta = load([path,'Theta.txt'] , '-ascii');
figure(3)
hold on;
plot3(theta(1, ~any(theta > 0, 1)), theta(2, ~any(theta > 0, 1)), theta(3, ~any(theta > 0, 1)), 'ob');
plot3(theta(1, any(theta > 0, 1)), theta(2, any(theta > 0, 1)), theta(3, any(theta > 0, 1)), 'xy');
plot3(theta(1, all(theta > 0, 1)), theta(2, all(theta > 0, 1)), theta(3, all(theta > 0, 1)), 'xr');

plotPlane([0, 0, 0], [1, 0, 0], -3:3:3, -3:3:3)
plotPlane([0, 0, 0], [0, 1, 0], -3:3:3, -3:3:3)
plotPlane([0, 0, 0], [0, 0, 1],  -3:3:3, -3:3:3)
plot3(-3, -3, -3, 'dm');
axis equal;

outliers = sum(any(theta > 0, 1));
acaso = sum(all(theta > 0, 1));
total = size(theta, 2);
ratio = (total-outliers) / total * 100;

disp('inlier ratio:')
disp(ratio)
disp('completely unstable parameters:')
disp(acaso)
disp('----------------')
disp(theta(:, all(theta > 0, 1)))

%% Plot features vectors
phi = load([path,'Phi.txt'] , '-ascii');
figure(4)
hold on;
inliers = all(phi > -1000);
plot3(phi(1, inliers), phi(2, inliers), phi(3, inliers), 'ob');

disp('----------------')
disp('inlier features ratio:')
disp(sum(inliers)/length(phi));

disp('outliers features:')
disp(length(phi)-sum(inliers))
