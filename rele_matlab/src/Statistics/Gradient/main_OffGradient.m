%% Script to read REPS statistics
addpath('..');

%clear old data
clear

%clear old figures
figure(1)
clf(1)


%% Read data

disp('Reading agent data...')
csv = csvread('/tmp/ReLe/Offpolicy/Deep/Deep_agentData.log');

disp('Organizing data...')

index = 1;
ep = 1;

while(index < size(csv, 1))
    [data(ep), index] = ReadOffGradientStatistics(csv, index); 
    ep = ep + 1;
end

clearvars csv

J = [];
N = [];
for i=1:size(data,2)
        J = [J; mean(data(i).J)];
        for k = 1:length(data(i).histGradient)
            N = [N; norm(data(i).histGradient(k).g,inf)];
        end
end

plot(N);
% plot(J);
axis tight
