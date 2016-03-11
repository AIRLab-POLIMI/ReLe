%% Plott all the deta from LQR test
close all
clear
clc

addpath(genpath('../..'));

%% load data
pathBase = '/tmp/ReLe/lqrPrint/';

baseline{1} = 'normal';
baseline{2} = 'trace'; 
%baseline{3} = 'diag'; 

numEpisodes = [100, 1000, 10000, 100000];

numTests = 2;


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
            %data.E(i)=E;
        end      
        
        % Compute mean
        results{indx1, indx2}.F  = mean(data.F, 1);
        results{indx1, indx2}.Fs = mean(data.Fs, 1);
        results{indx1, indx2}.G  = mean(data.G, 1);
        results{indx1, indx2}.T  = mean(data.T, 1);
        results{indx1, indx2}.J  = mean(data.J, 1);
        
        % Compute variance
        results{indx1, indx2}.covF  = diag(cov(data.F, 1));
        results{indx1, indx2}.covFs = diag(cov(data.Fs, 1));
        results{indx1, indx2}.covG  = diag(cov(data.G, 1));
        results{indx1, indx2}.covT  = diag(cov(data.T, 1));
        results{indx1, indx2}.covJ  = diag(cov(data.J, 1));
        
        % Update index
        indx2 = indx2+1;
    end
    % Update index
    indx1 = indx1+1;
end


%% plot data
for i = 1:length(baseline)
    for j = 1:length(numEpisodes)
        % get mean
        F = results{i, j}.F;
        Fs = results{i, j}.Fs;
        G = results{i, j}.G;
        T = results{i, j}.T;
        J = results{i, j}.J;
        
        %get covariance
        covF = results{i, j}.covF;
        covFs = results{i, j}.covFs;
        covG = results{i, j}.covG;
        covT = results{i, j}.covT;
        covJ = results{i, j}.covJ;
        
        %compute min
        [minF, indF] = min(F);
        [minFs, indFs] = min(Fs);
        [minG, indG] = min(G);
        
        % plot
        figure(i)
        subplot(4,3,1)
        hold on
        shadedErrorBar(1:length(F), F, 2*sqrt(covF), {'LineWidth', 2'}, 1);
        plot(indF, minF, 'dm')
        title('ExpectedDelta')
        axis tight
        
        subplot(4,3,2)
        hold on
        shadedErrorBar(1:length(Fs), Fs,2*sqrt(covFs), {'LineWidth', 2'}, 1);
        plot(indFs, minFs, 'dm')
        title('SignedExpectedDelta')
        axis tight
        
        subplot(4,3,3)
        hold on
        shadedErrorBar(1:length(G), G, 2*sqrt(covG), {'LineWidth', 2'}, 1);
        plot(indG, minG, 'dm')
        title('GradientNorm')
        axis tight
        
        subplot(4,3,4)
        shadedErrorBar(1:length(J), J, 2*sqrt(covJ),{'LineWidth', 2'}, 1);
        title('Objective Function')
        axis tight
        
        subplot(4,3,5)
        shadedErrorBar(1:length(T), T, 2*sqrt(covT),{'LineWidth', 2'}, 1);
        title('Trace of hessian')
        axis tight
        
    end
end

