%% Scripts to visualize GIRL data

addpath('../Statistics')

%% Load data
clc
clear
close all

path = '/tmp/ReLe/Graphs/';

mkdir(path)

load('/tmp/ReLe/G.txt', '-ascii');
load('/tmp/ReLe/J.txt', '-ascii');
load('/tmp/ReLe/D.txt', '-ascii');

gridPoints = size(G, 1);
stepSize = 0.02;
startValue = -5;

p = 0:gridPoints-1;
p = p * stepSize + startValue;

figure(1)
plot(p, G)
title('normG')
xlabel('p')
ylabel('|G|')
saveas(1, [path, 'G.jpg'])

figure(2)
plot(p, J)
title('J')
xlabel('p')
ylabel('J')
saveas(2, [path, 'J.jpg']);


figure(3)
plot(p, D*1000)
title('D')
xlabel('p')
ylabel('D')
saveas(3, [path, 'D.jpg']);

figure(4)
plot(p, G./J)
title('G/J')
xlabel('p')
ylabel('|G|/J')
saveas(4, [path, 'G_J.jpg']);

figure(5)
semilogy(p, G./(J.^2))
title('G/J2')
xlabel('p')
ylabel('|G|/J^2')
saveas(5, [path, 'G_J2.jpg']);
 
figure(6)
semilogy(p, G./D)
title('G/D')
xlabel('p')
ylabel('|G|/D')
saveas(6, [path, 'G_D.jpg']); 


%% debug figures
figure(7)
plot(p, gradient(G.^2)/stepSize)
title('dG2')
xlabel('p')
ylabel('dG2')

figure(8)
plot(p, gradient((G./D).^2)/stepSize)
title('d(G/D)^2')
xlabel('p')
ylabel('d(G/D)^2')

%% Trajectories
csv = csvread('/tmp/ReLe/Trajectories.txt');
traj = readDataset(csv);

figure(9)
hold on;
for episode = 1:size(traj,1)
    
    if(size(traj(episode).x, 2) == 1)
        plot(traj(episode).x);
    else if(size(traj(episode).x, 2) == 2)
            plot3(1:size(traj(episode).x,1), traj(episode).x(:, 1), traj(episode).x(:,2));
        end
    end
end