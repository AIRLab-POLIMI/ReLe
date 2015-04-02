%% Script to read REPS statistics
addpath('..');

%clear old data
clear

%clear old figures
figure(1)
clf(1)


%% Read data

disp('Reading agent data...')
csv = csvread('/tmp/ReLe/Portfolio/PG/Portfolio_agentData.log');

disp('Organizing data...')

index = 1;
ep = 1;

while(index < size(csv, 1))
    [data(ep), index] = ReadGradientStatistics(csv, index);
    ep = ep + 1;
end

clearvars csv

J = [];
N = [];
for i=1:size(data,2)
    J = [J; mean(data(i).J)];    
    N = [N; norm(data(i).gradient,2)];
end

plot(J);
% plot(J);
axis tight
