%% Rocky data visualizer
addpath('../Statistics');
addpath('../Toolbox/');

%clear old data
clear

% cmd = '/home/mpirotta/Projects/github/ReLe/rele-build/segway_BBO';
% status = system(cmd);

%% Read data

disp('Reading data trajectories...')
csv = csvread('/tmp/ReLe/mh/BBO/mhFinal.log');

disp('Organizing data in episodes...')
episodes = readDataset(csv);
clearvars csv

%% Display Data
close all
disp('Plotting trajectories...')
for ep = 1:length(episodes)
    close all;
    figure();
    plot(ep.x(1:end-1,2),ep.x(1:end-1,3))
    plot(ep.x(1,2), ep.x(1,3), 'sg')
    plot(ep.x(end-1,2), ep.x(end-1,3), 'or')
    legend('path', 'start', 'end')
    pause
end


