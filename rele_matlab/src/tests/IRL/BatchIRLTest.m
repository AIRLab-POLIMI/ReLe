%% Plott all the deta from LQR test
close all
clear
clc

addpath(genpath('../..'));

%% load data
pathBase = '/tmp/ReLe/lqrPrint/';
outPath = '/tmp/ReLe/matlab_out/lqrPrint/';

[~,~,~] = mkdir(outPath);

baseline{1} = 'normal';
baseline{2} = 'trace'; 
baseline{3} = 'diag'; 

numEpisodes = [100, 1000, 10000, 100000];

numTests = 10000;


indx1 = 1;
%% read batches of data
results = cell(length(baseline), length(numEpisodes));
for b = baseline
    indx2 = 1;
    for n = numEpisodes   
        for i = 1:numTests
            % generate path
            pathC = strcat(pathBase, b, '/', num2str(n), '/', num2str(i), '/');
            path = pathC{1};
            
            % load
            load([path, 'F.txt'] ,'ascii')
            load([path, 'Fs.txt'] ,'ascii')
            load([path, 'G.txt'] ,'ascii')
            load([path, 'T.txt'] ,'ascii')
            load([path, 'J.txt'] ,'ascii')
            load([path, 'E.txt'] ,'ascii')
            
            % store
            data.F(i, :)=F;
            data.Fs(i, :)=Fs;
            data.G(i, :)=G;
            data.T(i, :)=T;
            data.J(i, :)=J;
            data.E1(i, :)=E(:, 1);
            data.E2(i, :)=E(:, 2);
        end      
        
        % Compute mean
        results{indx1, indx2}.F  = mean(data.F, 1);
        results{indx1, indx2}.Fs = mean(data.Fs, 1);
        results{indx1, indx2}.G  = mean(data.G, 1);
        results{indx1, indx2}.T  = mean(data.T, 1);
        results{indx1, indx2}.J  = mean(data.J, 1);
        results{indx1, indx2}.E1  = mean(data.E1, 1);
        results{indx1, indx2}.E2  = mean(data.E2, 1);
        
        % Compute variance
        results{indx1, indx2}.covF  = diag(cov(data.F, 1));
        results{indx1, indx2}.covFs = diag(cov(data.Fs, 1));
        results{indx1, indx2}.covG  = diag(cov(data.G, 1));
        results{indx1, indx2}.covT  = diag(cov(data.T, 1));
        results{indx1, indx2}.covJ  = diag(cov(data.J, 1));
        results{indx1, indx2}.covE1 = diag(cov(data.E1, 1));
        results{indx1, indx2}.covE2 = diag(cov(data.E2, 1));
        
        % Update index
        indx2 = indx2+1;
    end
    % Update index
    indx1 = indx1+1;
end


%% Load exact data         
path = [pathBase, '/exact/'];
exact.F  = load([path, 'F.txt'] ,'ascii');
exact.Fs = load([path, 'Fs.txt'] ,'ascii');
exact.G  = load([path, 'G.txt'] ,'ascii');
exact.T  = load([path, 'T.txt'] ,'ascii');
exact.J  = load([path, 'J.txt'] ,'ascii');
Eexact = load([path, 'E.txt'] ,'ascii');
exact.E1  = Eexact(:, 1);
exact.E2  = Eexact(:, 2);

%% plot data
for i = 1:length(baseline)
    for j = 1:length(numEpisodes)
        % get mean
        F = results{i, j}.F;
        Fs = results{i, j}.Fs;
        G = results{i, j}.G;
        T = results{i, j}.T;
        J = results{i, j}.J;
        E1 = results{i, j}.E1;
        E2 = results{i, j}.E2;
        
        %get covariance
        covF = results{i, j}.covF;
        covFs = results{i, j}.covFs;
        covG = results{i, j}.covG;
        covT = results{i, j}.covT;
        covJ = results{i, j}.covJ;
        covE1 = results{i, j}.covE1;
        covE2 = results{i, j}.covE2;
        
        %compute min
        [minF, indF] = min(F);
        [minFs, indFs] = min(Fs);
        [minG, indG] = min(G);
        
        % plot
        figN = (i-1)*length(numEpisodes)+j;
        figure(figN)
        
        subplot(2,3,1)
        hold on
        shadedErrorBar(1:length(F), F, 2*sqrt(covF), {'LineWidth', 2'}, 1);
        plot(exact.F);
        plot(indF, minF, 'dm')
        title('ExpectedDelta')
        axis tight
        
        subplot(2,3,2)
        hold on
        shadedErrorBar(1:length(Fs), Fs,2*sqrt(covFs), {'LineWidth', 2'}, 1);
        plot(exact.Fs);
        plot(indFs, minFs, 'dm')
        title('SignedExpectedDelta')
        axis tight
        
        subplot(2,3,3)
        hold on
        shadedErrorBar(1:length(G), G, 2*sqrt(covG), {'LineWidth', 2'}, 1);
        plot(exact.G);
        plot(indG, minG, 'dm')
        title('GradientNorm')
        axis tight
        
        subplot(2,3,4)
        hold on
        shadedErrorBar(1:length(J), J, 2*sqrt(covJ),{'LineWidth', 2'}, 1);
        plot(exact.J);
        title('Objective Function')
        axis tight
        
        subplot(2,3,5)
        hold on
        shadedErrorBar(1:length(T), T, 2*sqrt(covT),{'LineWidth', 2'}, 1);
        plot(exact.T);
        title('Trace of hessian')
        axis tight
        
        subplot(2,3,6)
        hold on
        shadedErrorBar(1:length(E1), E1, 2*sqrt(covE1),{'LineWidth', 2'}, 1);
        shadedErrorBar(1:length(E2), E2, 2*sqrt(covE2),{'LineWidth', 2'}, 1);
        plot(exact.E1);
        plot(exact.E2);
        title('EigenValues')
        axis tight
        
        
       suptitle(strcat(baseline{i}, ' - ', num2str(numEpisodes(j))))
       
       savefig([outPath, num2str(figN),'.fig']);
    end
end

