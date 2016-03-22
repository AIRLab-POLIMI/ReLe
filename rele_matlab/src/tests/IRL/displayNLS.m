%% Print Trajectories
close all
clear

addpath(genpath('../../..'));

fraction = 0.1;
step = floor(1/fraction);

outPath = '/tmp/ReLe/matlab_out/nls/';
[~,~,~] = mkdir(outPath);

%% create cylinder
[yc, zc, xc] = cylinder();

xc = 80*xc;
yc = 0.1*yc;
zc = 0.1*zc;

%% list algorithms
alg = {
    %'GIRL', ...
    %'ExpectedDeltaIRL',  ...
    'EGIRL', ...
    'EMIRL', ...
    'EpisodicExpectedDeltaIRL'};

lastindex = length(alg);


%% Plot stuff
for i = 1:lastindex
    % plot trajectories of imitator
    csv = csvread(['/tmp/ReLe/nls/', alg{i}, '/TrajectoriesImitator.txt']);
    traj = readDataset(csv);
    
    figure(i)
    subplot(3, 1, 1);
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
    
    
    
    % plot trajectories of expert
    csv = csvread(['/tmp/ReLe/nls/', alg{i}, '/TrajectoriesExpert.txt']);
    traj = readDataset(csv);
    
    subplot(3, 1,2);
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
    
    
    % Plot recovered reward Function
    w = load(['/tmp/ReLe/nls/', alg{i}, '/Weights.txt'] , '-ascii');
    
    [X,Y] = meshgrid(-3:0.1:3);
    
    Z = zeros(size(X));
    
    for l = 1:length(X)
        for j = 1:length(X)
            Z(l, j) = basis_krbf(5,[-2, 2; -2, 2], [X(l, j); Y(l, j)])'*w;
        end
        
    end
    
    subplot(3, 1,3);
    mesh(X, Y, Z)
    title('weights');
    
    %% set title and save figure
    suptitle(alg{i})
    savefig([outPath, alg{i},'.fig']);
end