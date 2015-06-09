%% Rocky data visualizer
addpath('../Statistics');
addpath('../Toolbox/');

%clear old data
clear

% cmd = '/home/matteo/Projects/github/ReLe/rele-build/mh_BBO';
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
for k = 1:length(episodes)
    ep = episodes(k);
    close all;
    figure(); hold on;
    plot(ep.x(1:end-1,2),ep.x(1:end-1,3))
    plot(ep.x(1,2), ep.x(1,3), 'sg')
    plot(ep.x(end-1,2), ep.x(end-1,3), 'or')
    plot([17.5,17.5], [17.5,22], 'b')
    plot([17.5,22], [22,22], 'b')
    plot([22,22], [22,17.5], 'b')
    plot([17.5,22], [17.5,17.5], 'b')
    legend('path', 'start', 'end')
    
    figure(2);
    subplot(3,1,1); hold on;
    plot(1:length(ep.x(1:end-1,2)),ep.x(1:end-1,2));
    plot(1:length(ep.x(1:end-1,3)),22*ones(length(ep.x(1:end-1,3)),1));
    plot(1:length(ep.x(1:end-1,3)),17.5*ones(length(ep.x(1:end-1,3)),1));
    hold off;
    subplot(3,1,2); hold on;
    plot(1:length(ep.x(1:end-1,3)),ep.x(1:end-1,3));
    plot(1:length(ep.x(1:end-1,3)),22*ones(length(ep.x(1:end-1,3)),1));
    plot(1:length(ep.x(1:end-1,3)),17.5*ones(length(ep.x(1:end-1,3)),1));
    hold off;
    subplot(3,1,3);
    plot(1:length(ep.x(1:end-1,2)),ep.r(1:end-1));
    pause
end