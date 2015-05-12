%% Rocky data visualizer
addpath('../Statistics');

%clear old data
clear

%clear old figures
figure(1)
clf(1)

figure(2)
clf(2)
%% Choose file
hpg = false;
if(hpg)
    basedir = '/tmp/ReLe/Rocky/HPG/';
else
    basedir = '/tmp/ReLe/Rocky/REPS/';
end
trajectoryFile = [basedir 'Rocky.log'];
agentFile = [basedir 'Rocky_agentData.log'];

%% Read data

disp('Reading data trajectories...')
csv = csvread(trajectoryFile);

disp('Organizing data in episodes...')
episodes = readDataset(csv);
clearvars csv

%% Plot J
if(hpg)
    plotGradient(1, agentFile);
else
    plotREPS(1, agentFile);
end


%% Display Data

disp('Plotting trajectories...')


for i=1:100:size(episodes, 1)
    x = episodes(i).x;
    
    traj = x(:, 1:2);
    rockytraj = traj + x(:, 6:7);

    figure(2)
    hold on;
    plot(traj(:, 1), traj(:, 2), 'b');
    plot(rockytraj(:, 1), rockytraj(:, 2), 'm');
end

figure(2)
axis equal

disp('Starting visualization...')

figure(3)
hold on
plot(traj(:, 1), traj(:, 2), 'b');
plot(rockytraj(:, 1), rockytraj(:, 2), 'm');
axis equal;
lim = axis;
pause(2)
clf(3)
figure(3)
H = uicontrol('Style', 'PushButton', ...
                    'String', 'Break', ...
                    'Callback', 'delete(gcbf)');

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
   plot(traj(i, 1), traj(i, 2), 'xb');
   plot(rockytraj(startIndex:i, 1), rockytraj(startIndex:i, 2), 'm');
   plot(rockytraj(i, 1), rockytraj(i, 2), 'xm');
   axis(lim);
   pause(0.01)
end



