%% Script to read REPS statistics
addpath('..');

%clear old data
clear

%clear old figures
figure(1)
clf(1)


%% Read data

disp('Reading agent data...')
csv = csvread('/tmp/ReLe/Rocky/REPS/Rocky_agentData.log');

disp('Organizing data...')

index = 1;
ep = 1;

while(index < size(csv, 1))
    [data(ep), index] = ReadREPSStatistics(csv, index); 
    ep = ep + 1;
end

clearvars csv

J = [];
for i=1:size(data,2)
    for j=1:size(data(i).policies, 2)
        J = [J; data(i).policies(j).J];
    end
end

plot(J);
axis tight
