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

titles{1} = 'Imitator 1 - Hessian';
titles{2} = 'Imitator 2 - Sparse';
lastindex = 2;


for i = 1:lastindex

    csv = csvread(['/tmp/ReLe/nls/PGIRL/TrajectoriesImitator', num2str(i-1) '.txt']);
    traj = readDataset(csv);

    figure(i)
    title(titles{i})
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

end

%% plot trajectories of expert
csv = csvread('/tmp/ReLe/nls/PGIRL/TrajectoriesExpert.txt');
traj = readDataset(csv);

figure(lastindex + 1)
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


%% Plot recovered reward Function
W = load('/tmp/ReLe/nls/PGIRL/Weights.txt' , '-ascii');

for k = 1:lastindex
    
w = W(:, k);

[X,Y] = meshgrid(-1:0.1:2);

Z = zeros(size(X));

for i = 1:length(X)
    for j = 1:length(X)
        Z(i, j) = w'*basis_krbf(5,[-1, 2; -1, 2], [X(i, j); Y(i, j)]);
    end

end

figure(lastindex + 1 +k)
mesh(X, Y, Z)
title(titles{k})
end