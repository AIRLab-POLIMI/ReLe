%% Plot the gradient

%% Load data
clc
clear
close all

normG2 = load('/tmp/ReLe/normG.txt', '-ascii');
normJ2 = load('/tmp/ReLe/normJ.txt', '-ascii');

normG = sqrt(normG2);

gridPoints = size(normG, 1);
stepSize = 0.02;
startValue = -10.0;

p = 0:gridPoints-1;
p = p * stepSize + startValue;

figure(1)
plot(p, normG)
title('normG')
xlabel('p')
ylabel('|g|')
saveas(1, 'normG.jpg')

figure(2)
plot(p, normJ2)
title('normJ')
xlabel('p')
ylabel('|J|^2')
saveas(2, 'normJ2.jpg');

figure(3)
plot(p, normG./(normJ2))
title('normGN')
xlabel('p')
ylabel('|g|/|J|^2')
saveas(3, 'normGN.jpg');

figure(4)
plot(p, log(normG) - log(normJ2))
title('normGNlog')
xlabel('p')
ylabel('log|g| - 2log|J|')
saveas(4, 'normGNlog.jpg');

figure(5)
plot(p, stepSize*gradient(log(normG) - log(normJ2)))