%% Print Trajectories
close all
clear all

addpath('../../iodata')

fraction = 0.1;
step = floor(1/fraction);

%% create cylinder
[yc, zc, xc] = cylinder();

xc = 80*xc;
yc = 0.1*yc;
zc = 0.1*zc;

%% plot trajectories of imitator

titles{1} = 'Imitator 1 -Normalization: None';
titles{2} = 'Imitator 2 -Normalization: Log Disparity';
lastindex = 2;


for i = 1:lastindex

    csv = csvread(['/tmp/ReLe/nls/GIRL/TrajectoriesImitator', num2str(i-1) '.txt']);
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
csv = csvread('/tmp/ReLe/nls/GIRL/TrajectoriesExpert.txt');
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