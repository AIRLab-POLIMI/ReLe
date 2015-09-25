%% Print Trajectories
close all
clear all

addpath(genpath('../../..'));

fraction = 0.1;
step = floor(1/fraction);

%% plot trajectories of expert
csv = csvread('/tmp/ReLe/lqr/EMIRL/TrajectoriesExpert.txt');
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

axis([-10, 10, -10, 10])
zlim([-10 10])

%% plot Parameters of expert
theta = load('/tmp/ReLe/lqr/EMIRL/Theta.txt' , '-ascii');
figure(3)
plot3(theta(1, :), theta(2, :), theta(3, :), 'o');
axis equal;

