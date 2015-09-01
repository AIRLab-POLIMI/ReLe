%% ShipSteering data visualizer
addpath('../Statistics');

%clear old data
clear

%clear old figures
close all

%% Choose file
basedir = '/tmp/ReLe/ShipSteering/PG/';
trajectoryFile = [basedir 'ship_r.log'];
gradientFile = [basedir 'ship_r_agentData.log'];

%% Read data

disp('Reading data trajectories...')
csv = csvread(trajectoryFile);

disp('Organizing data in episodes...')
episodes = readDataset(csv);
clearvars csv

gate = [100, 120; 120, 100];

%% Plot J
plotGradient(1, gradientFile);

%% Display Trajectories

disp('Plotting trajectories...')


for i=1:100:size(episodes, 1)
    x = episodes(i).x;
    
    traj = x(:, 1:2);
    
    figure(2)
    hold on;
    plot(traj(:, 1), traj(:, 2), 'b');
end

figure(2)
axis equal

%% Visualize last trajectory
disp('Starting visualization...')

figure(3)
hold on
plot(traj(:, 1), traj(:, 2), 'b');
plot(gate(:, 1), gate(:, 2), 'x');
axis auto;
lim = axis;
pause(2)
clf(3)
figure(3)
H = uicontrol('Style', 'PushButton', ...
                    'String', 'Break', ...
                    'Callback', 'delete(gcbf)');

historyToShow = 30;
for i = 1:size(traj, 1)
    if(~ishandle(H))
        break
    end
   startIndex = max(1, i-historyToShow);
   figure(3)
   hold off
   plot(traj(startIndex:i, 1), traj(startIndex:i, 2), 'b');
   hold on
   plot(traj(i, 1), traj(i, 2), 'xb');
   plot(gate(:, 1), gate(:, 2), 'x');
   axis(lim);
   pause(0.01)
end
