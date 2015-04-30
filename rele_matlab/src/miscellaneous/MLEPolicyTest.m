addpath('MLE');
addpath('../Statistics/');
excmd = '../../../rele-build/mle_Policy';
nbEp = 150;

status = system([excmd, ' ', num2str(nbEp), ' 0.5 0.5']);


disp('Reading data trajectories...')
csv = csvread('/tmp/ReLe/mle_Policy/test/data.log');

disp('Organizing data in episodes...')
episodes = readDataset(csv);
clearvars csv

DSS = []; DSA = [];
for i = 1:length(episodes)
    DSS = [DSS; episodes(i).x(1:end-1)];
    DSA = [DSA; episodes(i).u(1:end-1)];
end


custpdf = @(data,w,s) linearStateNormal(data,w,s,DSS);
samples = DSA;
phat = mle(samples, 'pdf', @(data,w,s) custpdf(data,w,s), 'start', [0;6])
