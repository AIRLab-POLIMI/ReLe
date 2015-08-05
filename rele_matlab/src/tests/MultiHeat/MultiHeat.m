%% Rocky data visualizer
addpath('../Statistics');
addpath(genpath('../Toolbox/'));

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

k = randi(length(episodes));

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
subplot(4,1,1); hold on;
plot(1:length(ep.x(1:end-1,2)),ep.x(1:end-1,2));
plot(1:length(ep.x(1:end-1,3)),22*ones(length(ep.x(1:end-1,3)),1));
plot(1:length(ep.x(1:end-1,3)),17.5*ones(length(ep.x(1:end-1,3)),1));
hold off;
subplot(4,1,2); hold on;
plot(1:length(ep.x(1:end-1,3)),ep.x(1:end-1,3));
plot(1:length(ep.x(1:end-1,3)),22*ones(length(ep.x(1:end-1,3)),1));
plot(1:length(ep.x(1:end-1,3)),17.5*ones(length(ep.x(1:end-1,3)),1));
hold off;
subplot(4,1,3);
plot(1:length(ep.x(1:end-1,2)),ep.r(1:end-1));
subplot(4,1,4);
plot(1:length(ep.u(1:end-1)),ep.u(1:end-1));


figure(3); clf; hold on;
title('All Modes');
for k=1:length(episodes)
	ep = episodes(k);
	
	col = flipud(lines(3));
	for u=1:3
		idx = ep.u(:) == mod(u,3);
		plot(ep.x(idx,2),ep.x(idx,3),'.','Color',col(u,:));
	end
end
plot([17.5,17.5], [17.5,22], 'k','LineWidth',2);
plot([17.5,22], [22,22], 'k','LineWidth',2);
plot([22,22], [22,17.5], 'k','LineWidth',2);
plot([17.5,22], [17.5,17.5], 'k','LineWidth',2);
grid on; box on; axis equal;
xlim([13.5 26]); ylim(xlim);


%% plot policy
%clear all;
csv = csvread(['/tmp/ReLe/mh/BBO/deep_nes_diag_agentData.log']);
index = 1;
ep = 1;
while(index < size(csv, 1))
    [data(ep), index] = ReadNESStatistics(csv, index);
    %     [data(ep), index] = ReadREPSStatistics(csv, index);
    ep = ep + 1;
end

nbparams = multiheat_basis(); %numero di parametri
theta = data(end).params(1:nbparams); %leggi parametri
policy = gibbs(@multiheat_basis, theta, [0,1,2]);


%% 

N = 1e3;
figure(10); clf; hold on;
states = (26-13.5)*rand(N,2) + 13.5;
col = flipud(lines(3));
for k=1:size(states,1)
	state = states(k,:);
	
% 	policy.probability(state',action)
	u = policy.drawAction(state');
	plot(state(1),state(2),'.','Color',col(u,:));
end
plot([17.5,17.5], [17.5,22], 'k','LineWidth',2);
plot([17.5,22], [22,22], 'k','LineWidth',2);
plot([22,22], [22,17.5], 'k','LineWidth',2);
plot([17.5,22], [17.5,17.5], 'k','LineWidth',2);
grid on; box on; axis equal;
xlim([13.5 26]); ylim(xlim);
