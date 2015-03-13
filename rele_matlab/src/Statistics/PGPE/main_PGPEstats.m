fn = '/home/matteo/Projects/github/ReLe/rele-build/prova.txt';
fp = fopen(fn,'r');
% stats = ReadPGPEStatistics(fp);
i = 1;
while ~feof(fp)
stats(i) = ReadPGPEIterationStats(fp)
i = i + 1;
end
stats
fclose(fp);

% %%
% fn1 = '/home/matteo/Projects/github/ReLe/rele-build/prova.txt';
% A = dlmread(fn1);
% data = ReadEpisodeBasedDataset(A)
% 
% sum(data.r .* (0.99*ones(1,length(data.r))).^(0:length(data.r)-1))


close all
stats(:).metaParams
plot(1:length([stats(:).metaParams]), [stats(:).metaParams])


% Jrho = zeros(length(stats),1);
% for i = 1:length(stats)
%     v = [stats(i).policies(:).J];
% %     if mean(v) > 0
% %         disp(mean(v));
% %     end
%     Jrho(i) = mean(v);
% end
% plot(1:length(Jrho), Jrho);
%%
clc
dim = 1;
syms a 
sigma = sqrt(0.01);
mu    = sym('w',   [dim,  1]);
dist = 1/(sqrt(2*pi) * (sigma)) * exp(-0.5*(a-mu)^2/(sigma)^2);
% pretty(pol)
% eval(subs(pol, [w; k; phi; a], [wnum; knum; state; action]))
g = gradient(log(dist), mu);

PREC = 0.00001;

it = length(stats);
grad = zeros(dim,it);
for i = 1:it
    nbpol = length(stats(i).policies);
    for p = 1:nbpol
        nbep = length(stats(i).policies(p).J);
        J = zeros(nbep,1);
        for ep = 1:nbep
%             fn1 = ['/home/matteo/Projects/github/ReLe/rele-build/'...
%                 'PGPE_tracelog_r' num2str(i-1) '_p' num2str(p-1) ...
%                 '_e' num2str(ep-1) '.txt'];
%             A = dlmread(fn1);
%             data = ReadEpisodeBasedDataset(A);
%             J(ep) = sum(data.r .* (0.99*ones(1,length(data.r))).^(0:length(data.r)-1));
%             assert(abs(J(ep)-stats(i).policies(p).J(ep))<=abs(J(ep))*PREC);
            J(ep) = stats(i).policies(p).J(ep);
        end
        J = mean(J);
        diffdist = eval(subs(g, [a, mu], [stats(i).policies(p).policy, stats(i).metaParams]));
        dl = stats(i).policies(p).difflog;
        assert(max(abs(diffdist-dl))<=0.1);
        grad(:,i) = grad(:,i) + diffdist * J;
    end
    grad(:,i) = grad(:,i)/nbpol;
    disp([grad(i), stats(i).metaGradient])
    [v,id] = max(abs(grad(i)-stats(i).metaGradient));
    assert(v<=abs(grad(id))*PREC);
    fprintf('#IT %d: OK!!\n', i); 
end

%%
A = rand(3,3);
b = rand(3,1);
inv(A)*b



