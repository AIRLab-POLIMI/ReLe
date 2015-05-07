%% Mountain Car data visualizer
clc; clear all;
addpath('../Statistics');

%% Read data

disp('Reading data trajectories...')
csv = csvread('/tmp/ReLe/mc/GIRL/data.log');
% csv = csvread('/tmp/ReLe/mc/GIRL/mletraining.log');

disp('Organizing data in episodes...')
episodes = readDataset(csv);
clearvars csv

%% Draw hill
% Rasmussen, Kuss, Gaussian Processes in Reinforcement Learning, NIPS 2003

clc; close all;
figure(1);
x = -1.3:0.05:1;
h = Hill(x);
% plot(x,h);
% hold on;

for i = 1:length(episodes)
    for t = 1:size(episodes(i).x,1)
        s  = episodes(i).x(t,2);
        hs = Hill(s);
        
        plot(x,h);
        title(['Episode ', num2str(i)]);
        line([0.5 0.5],[-0.3 0.5], 'Color', [.8 .8 .8]);
        ylim([-0.3, 0.5]);
        hold on;
        plot(s,hs,'ob', 'MarkerFace','b','MarkerSize', 10);
        hold off;
        
        pause(0.1);
    end
end

%% Read mle data

disp('Reading data trajectories...')
csv = csvread('/tmp/ReLe/mc/GIRL/mledata.log');

disp('Organizing data in episodes...')
mleEpisodes = readDataset(csv);
clearvars csv

%% compare trajectories

idx = 1;
figure(2);
subplot(4,1,1); hold on;
plot(episodes(idx).x(:,2));
plot(mleEpisodes(idx).x(:,2));
legend('hand','mle');
ylabel('position');
grid on; hold off;

subplot(4,1,2); hold on;
plot(episodes(idx).x(:,1));
plot(mleEpisodes(idx).x(:,1));
legend('hand','mle');
ylabel('velocity');
grid on; hold off;

subplot(4,1,3); hold on;
plot(episodes(idx).u(:));
plot(mleEpisodes(idx).u(:));
legend('hand','mle');
ylabel('actions');
grid on; hold off;

subplot(4,1,4); hold on;
plot(episodes(idx).r(:));
plot(mleEpisodes(idx).r(:));
legend('hand','mle');
ylabel('reward');
grid on; hold off;