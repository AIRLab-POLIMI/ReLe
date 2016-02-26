%% Plott all the deta from LQR test
close all
clear
clc

%% load data
path = '/tmp/ReLe/';

load([path, 'F.txt'] ,'ascii')
load([path, 'Fs.txt'] ,'ascii')
load([path, 'G.txt'] ,'ascii')
load([path, 'T.txt'] ,'ascii')
load([path, 'J.txt'] ,'ascii')
load([path, 'E.txt'] ,'ascii')


%% plot data
figure(1)
subplot(2,3,1)
plot(F)
title('ExpectedDelta')
axis tight

subplot(2,3,2)
plot(Fs)
title('SignedExpectedDelta')
axis tight

subplot(2,3,3)
plot(G)
title('GradientNorm')
axis tight

subplot(2,3,4)
plot(J)
title('Objective Function')
axis tight

subplot(2,3,5)
plot(T)
title('Trace of hessian')
axis tight

subplot(2,3,6)
hold on
plot(E(:, 1))
plot(E(:, 2))
title('Eigenvalues of Hessian')
axis tight

