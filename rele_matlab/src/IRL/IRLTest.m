%% Plot the gradient

%% Load data
clear
close all

normG = load('/tmp/ReLe/norm.txt', '-ascii')
dNormG = load('/tmp/ReLe/gradient.txt', '-ascii')

gridPoints = size(normG, 1);
stepSize = 0.01;

p = -floor(gridPoints/2):1:floor(gridPoints/2);
p = p * stepSize;

figure(1)
hold on
plot(p, dNormG)
plot(p, gradient(normG)/stepSize)
title('gradient')
xlabel('p')
ylabel('d|g|')
legend('analytical', 'finite differences') 

figure(2)
plot(p, normG)
title('norm')
xlabel('p')
ylabel('|g|')
