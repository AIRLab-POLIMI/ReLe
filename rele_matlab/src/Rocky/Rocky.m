%% Rocky data visualizer
addpath('../Statistics');

%clear old data
clear

%clear old figures
figure(1)
clf(1)

figure(2)
clf(2)

figure(3)
clf(3)

%% Read data

disp('Reading data trajectories...')
csv = csvread('/home/dave/prova.txt');

disp('Organizing data in episodes...')
episodes = readDataset(csv);

%% 

disp('Plotting trajectories...')

r = zeros(size(episodes, 1), 1);
for i=1:size(episodes, 1)
    x = episodes{i, 1};
    xn = episodes{i, 3};
    r(i) = sum(episodes{i, 4})/ size(episodes{i, 4}, 1);
    
    traj = [x(:, 1:2); xn(end, 1:2)];
    rockytraj = traj + [x(:, 6:7); xn(end, 6:7)];

    figure(1)
    hold on;
    plot(traj(:, 1), traj(:, 2), 'b');
    plot(rockytraj(:, 1), rockytraj(:, 2), 'm');
end

figure(1)
axis equal

disp('Plotting mean reward...')

figure(2)
plot(r);
axis tight

disp('Starting visualization...')

figure(3)
plot(traj(:, 1), traj(:, 2), 'b');
plot(rockytraj(:, 1), rockytraj(:, 2), 'm');
axis auto;
lim = axis;
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



