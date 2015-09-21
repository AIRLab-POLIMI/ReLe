%% Pursuer data visualizer
addpath(genpath('../..'));

%clear old data
clear

%clear old figures
close all

%% Choose file
%gradientType = 'gb';
gradientType = 'r';

basedir = '/tmp/ReLe/pursuer/PG/';
trajectoryFile = [basedir 'pursuer_', gradientType, '.log'];
gradientFile = [basedir 'pursuer_', gradientType, '_agentData.log'];

%% Read data

disp('Reading data trajectories...')
csv = csvread(trajectoryFile);

disp('Organizing data in episodes...')
episodes = readDataset(csv);
clearvars csv

gate = [100, 120; 120, 100];

%% Plot J
plotGradient(1, gradientFile);

%% Display Data

disp('Plotting trajectories...')


for i=1:100:size(episodes, 1)
    x = episodes(i).x;
    
    traj = x(:, 1:2);
    traj = [traj, x(:, 3)];
    pursuerTraj = x(:, 1:2) + x(:, 4:5);
    pursuerTraj = [pursuerTraj, x(:, 6)];

    figure(2)
    hold on;
    plot(traj(:, 1), traj(:, 2), 'b');
    plot(pursuerTraj(:, 1), pursuerTraj(:, 2), 'm');
end

figure(2)
axis equal

disp('Starting visualization...')

figure(3)
hold on
plot(traj(:, 1), traj(:, 2), 'b');
plot(pursuerTraj(:, 1), pursuerTraj(:, 2), 'm');
axis equal;
lim = axis;
[sX, sY] = computeFraction(lim, 100);

pause(2)
clf(3)
figure(3)
H = uicontrol('Style', 'PushButton', ...
                    'String', 'Break', ...
                    'Callback', 'delete(gcbf)');
                
hold on

historyToShow = 100;
for i = 1:size(traj, 1)
    if(~ishandle(H))
        break
    end
   startIndex = max(1, i-historyToShow);
   figure(3)
   hold off
   plot(traj(startIndex:i, 1), traj(startIndex:i, 2), 'b');
   
   hold on
   plot2DPose(traj(i, 1), traj(i, 2), traj(i, 3), sX, sY, 'b');
   plot(pursuerTraj(startIndex:i, 1), pursuerTraj(startIndex:i, 2), 'm');
   plotCircle(pursuerTraj(i, 1), pursuerTraj(i, 2), 0.05);
   plot2DPose(pursuerTraj(i, 1), pursuerTraj(i, 2), pursuerTraj(i, 3), sX, sY, 'm');
   axis(lim);
   pause(0.01)
end



