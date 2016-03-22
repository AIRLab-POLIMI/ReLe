%% Print Trajectories
close all
clear
clc

addpath(genpath('../../..'));

fraction = 1;
step = floor(1/fraction);

outPath = '/tmp/ReLe/matlab_out/lqr/';
[~,~,~] = mkdir(outPath);

%% list algorithms

alg = {
    'EGIRL', ...
    'EMIRL', ...
    'EpisodicExpectedDeltaIRL'};

lastindex = length(alg);

%% plot trajectories of expert
for i = 1:lastindex
    figure(i)
    path = ['/tmp/ReLe/lqr/', alg{i}, '/'];
    csv = csvread([path, '/TrajectoriesExpert.txt']);
    traj = readDataset(csv);

    subplot(3, 1, 1);
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
    title('Expert Parameters')
    theta = load([path,'Theta.txt'] , '-ascii');
    subplot(3, 1, 2);
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

    %% Plot features vectors
    phi = load([path,'Phi.txt'] , '-ascii');
    subplot(3, 1, 3);
    title('Features')
    hold on;
    inliers = all(phi > -1000);
    plot3(phi(1, inliers), phi(2, inliers), phi(3, inliers), 'ob');

    disp('----------------')
    disp('inlier features ratio:')
    disp(sum(inliers)/length(phi));

    disp('outliers features:')
    disp(length(phi)-sum(inliers))
    
    %% set title and save figure
    suptitle(alg{i})
    savefig([outPath, alg{i},'.fig']);   
end

 %% Print weights
 figure(100)
 hold on;
 
 for i = 1:lastindex
    path = ['/tmp/ReLe/lqr/', alg{i}, '/'];
    omega = load([path,'Weights.txt'], '-ascii');  
    plot3(omega(1), omega(2), omega(3), '.b')
    text(omega(1), omega(2), omega(3), alg{i})
 end

% print real reward parameters
omega = [0.2 0.7 0.1];
plot3(omega(1), omega(2), omega(3), '.r');
text(omega(1), omega(2), omega(3), 'Expert');

title('Recovered Weights')
savefig([outPath, 'Weights','.fig']);
