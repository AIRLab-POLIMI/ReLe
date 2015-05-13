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
% csv = csvread('/tmp/ReLe/mc/GIRL/lspidata.log');

disp('Organizing data in episodes...')
mleEpisodes = readDataset(csv);
clearvars csv

%% compare trajectories

for idx = 1:20
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
    pause
    close all
end

%% load learned reward
close all;
for i = 0%:2
    figure(i+1);
    rewa{i+1} = dlmread(['/tmp/ReLe/mc/GIRL/learnedRew_action',num2str(i),'.log']);
    M = rewa{i+1};
    plot3(M(:,1), M(:,2), M(:,3), 'o');
    xlabel('velocity');
    ylabel('position');
    zlabel('reward');
%     xlim([-0.07,0.07]);
%     ylim([-1.2,0.6]);
    title(['action ', num2str(i)]);
end

rw = dlmread('/tmp/ReLe/mc/GIRL/rewards.log');

%% load gradients and gram matrix
grads = dlmread('/tmp/ReLe/lqr/GIRL/grad.log');
Gr = grads' * grads;

X = Gr(1:end-1,:) - repmat(Gr(end,:),size(Gr,1)-1,1);
Xc = dlmread('/tmp/ReLe/lqr/GIRL/X.log');
assert(max(max(abs(X - Xc))) <= 1e-8);
ns = null(X)

[U,S,V] = svd(X,0);
Uc = dlmread('/tmp/ReLe/lqr/GIRL/X.log');
Sc = dlmread('/tmp/ReLe/lqr/GIRL/s.log');
Vc = dlmread('/tmp/ReLe/lqr/GIRL/V.log');
Y = dlmread('/tmp/ReLe/lqr/GIRL/Y.log');

[Q,R] = qr(X');
Q = dlmread('/tmp/ReLe/lqr/GIRL/Q.log');
R = dlmread('/tmp/ReLe/lqr/GIRL/R.log');



%% load reward basis
A = dlmread('/tmp/ReLe/mc/GIRL/basis.txt');
% [X,Y] = meshgrid(-0.07:0.005:0.07, -1.2:0.05:0.6);
[X,Y] = meshgrid(-0.2:0.01:0.2, -3:0.1:2);
V = [X(:),Y(:)];
close all;
figure(10);
hold on;
xlabel('velocity');
ylabel('position');
for i = 1:size(A,1)
    basis{i} = MCRewBasis(V,[A(i,1), A(i,2)], [A(i,3), A(i,4)]);
    surf(X,Y, reshape(basis{i},size(X,1), size(X,2)));
end
hold off;


%% Read random data
disp('Reading data trajectories...')
csv = csvread('/tmp/ReLe/mc/GIRL/lspidata.log');

disp('Organizing data in episodes...')
lspiEpisodes = readDataset(csv);
clearvars csv

X =[];
for i = 1:length(lspiEpisodes)
    X = [X;lspiEpisodes(i).u(1:end-1)];
end
length(find(X==0))
length(find(X==1))
length(find(X==2))

X =[];
for i = 1:length(lspiEpisodes)
    X = [X;lspiEpisodes(i).x(1:end-1,:),lspiEpisodes(i).r(1:end-1)];
end
plot3(X(:,1),X(:,2),X(:,3),'o')
xlabel('velocity');
ylabel('position');

%% load lspi q-function approximation
close all;
for i = 0:2
    figure(i+1);
    rewa{i+1} = dlmread(['/tmp/ReLe/mc/GIRL/learnedQ_a',num2str(i),'.log']);
    M = rewa{i+1};
    plot3(M(:,1), M(:,2), M(:,3), 'o');
    xlabel('velocity');
    ylabel('position');
    zlabel('reward');
    xlim([-0.07,0.07]);
    ylim([-1.2,0.6]);
    title(['action ', num2str(i)]);
end
%% Read final data

disp('Reading data trajectories...')
csv = csvread('/tmp/ReLe/mc/GIRL/finaldata.log');

disp('Organizing data in episodes...')
finalEpisodes = readDataset(csv);
clearvars csv

X = [];
for i = 1:length(finalEpisodes)
    X = [X; length(finalEpisodes(i).x(1:end-1))];
end
mean(X)
std(X)
median(X)

MCDraw(finalEpisodes);

%% Read LSPI policy
% close all;
figure(9)
POL = dlmread('/tmp/ReLe/mc/GIRL/finalpol.log');
plot3(POL(:,1), POL(:,2), POL(:,3), 'o');
xlabel('velocity');
ylabel('position');
zlabel('action');