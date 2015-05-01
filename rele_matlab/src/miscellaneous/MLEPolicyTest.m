% clc; clear all;
addpath('MLE');
addpath('../Statistics/');
addpath(genpath('~/Dropbox/IRL_matlab/'));
excmd = '../../../rele-build/mle_Policy';
mkdir('/tmp/ReLe/mle_Policy/test');
nbEp = 150;

reward = [0.3;0.7];
rewardFile = '/tmp/ReLe/mle_Policy/test/reward.dat';
dlmwrite(rewardFile, reward, '\t');

startFile = '/tmp/ReLe/mle_Policy/test/start.dat';
polpFile = '/tmp/ReLe/mle_Policy/test/polp.dat';

polType = 'mvndiag';
startp = [0;6];
polp = [-0.7; 2.3];
polType = 'mvnrbf';
startp = [zeros(5,1)];
polp = [-2+2*rand(5,1)];
% polType = 'mvnpoly';
% startp = [zeros(length(reward),1)];

dlmwrite(startFile, startp, '\t');
dlmwrite(polpFile, polp, '\t');


scmd = [excmd, ' ', polType, ' ', startFile, ' ', num2str(nbEp), ...
    ' ', rewardFile, ' ', polpFile];
status = system(scmd);


% disp('Reading data trajectories...')
% csv = csvread('/tmp/ReLe/mle_Policy/test/data.log');
% 
% disp('Organizing data in episodes...')
% episodes = readDataset(csv);
% clearvars csv
% 
% DSS = []; DSA = [];
% for i = 1:length(episodes)
%     DSS = [DSS; episodes(i).x(1:end-1)];
%     DSA = [DSA; episodes(i).u(1:end-1)];
% end

mlecpp = dlmread('/tmp/ReLe/mle_Policy/test/mleparams.log');
[polp, mlecpp]

assert(max(abs(polp-mlecpp)) <= 0.1);

% custpdf = @(data,w,s) linearStateNormal(data,w,s,DSS);
% samples = DSA;
% if strcmp(polType,'mvndiag')
%     phat = mle(samples, 'pdf', @(data,w,s) custpdf(data,w,s), 'start', startp);
% elseif strcmp(polType, 'mvnrbf')
%     basis = @rbf_basis_lqr;
%     basisval = basis(DSS);
%     phat = mle(samples, 'pdf', @(data,w1,w2,w3,w4,w5) rbfNormal(basisval, data,w1,w2,w3,w4,w5), 'start', startp)
% end
% 
% mlecpp = dlmread('/tmp/ReLe/mle_Policy/test/mleparams.log');
% [phat, mlecpp]
% 
% assert(max(abs(phat-mlecpp))<=1e-3);
