%% Plot the gradient

%% Load data
clear
close all

normG = load('/tmp/ReLe/normG.txt', '-ascii');
normJ = load('/tmp/ReLe/normJ.txt', '-ascii');

gridPoints = size(normG, 1);
stepSize = 0.01;

p = 0:gridPoints-1;
p = p * stepSize;

figure(1)
plot(p, normG)
title('normG')
xlabel('p')
ylabel('|g|^2')

figure(2)
plot(p, normJ)
title('normJ')
xlabel('p')
ylabel('|R|')

figure(3)
plot(p, normG./(normJ.^2))
title('normGN')
xlabel('p')
ylabel('|g|^2/|R|^2')

figure(4)
plot(p, log(normG) - log(normJ.^2))
title('normGNlog')
xlabel('p')
ylabel('2log|g| - 2log|R|')