%% Plott all the deta from LQR test
close all
clear
clc

addpath(genpath('../..'));

%% load data
%path = '/tmp/ReLe/lqrPrint/';
path = '/tmp/ReLe/lqrPrint/exact/';

load([path, 'F.txt'] ,'ascii')
load([path, 'Fs.txt'] ,'ascii')
load([path, 'G.txt'] ,'ascii')
load([path, 'T.txt'] ,'ascii')
load([path, 'J.txt'] ,'ascii')
load([path, 'E.txt'] ,'ascii')

%% Find optimal values
[minF, indF] = min(F);
[minFs, indFs] = min(Fs);
[minG, indG] = min(G);


%% plot data
figure(1)
subplot(2,3,1)
hold on
plot(F)
plot(indF, minF, 'dm')
title('ExpectedDelta')
axis tight

subplot(2,3,2)
hold on
plot(Fs)
plot(indFs, minFs, 'dm')
title('SignedExpectedDelta')
axis tight

subplot(2,3,3)
hold on
plot(G)
plot(indG, minG, 'dm')
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

%% Plot trajectories
% csv = csvread([path, 'Trajectories.txt']);
% traj = readDataset(csv);
% subplot(4, 3, 7:12)
% 
% title('Trajectories')
% xlabel('t')
% ylabel('x1')
% zlabel('x2')
% axis tight
% 
% hold on;
% 
% 
% hop = 5;
% for episode = 1:hop:size(traj,1)
%     
%     if(size(traj(episode).x, 2) == 1)
%         plot(traj(episode).x);
%     else if(size(traj(episode).x, 2) == 2)
%             plot3(1:size(traj(episode).x,1), traj(episode).x(:, 1), traj(episode).x(:,2));
%         end
%     end
% end


%% print weight
disp((indF-1)/100)
disp((indFs-1)/100)
disp((indG-1)/100)

