%% TEST1.m
% Tests for NIPS
%  PGIRL with 10d and 20d lqr
%  100 weights
%  10, 50, 100, 500, 1000, 2000 ep

clear all; close all; clc;

rng('shuffle');
addpath('../../../../miscellaneous/');

gtypes{1} = 'r';
gtypes{2} = 'rb';
gtypes{3} = 'g';
gtypes{4} = 'gb';
gtypes{5} = 'enac';

nbWEIGHTS = 100;

tmpFolder = '/tmp/ReLe/lqr/GIRLNIPS_TEST2/';
cmd = '/home/mpirotta/Projects/github/ReLe/rele-build/lqr_GIRLNIPS_TEST2';

mkdir('tmpTEST2');

tmpPrefix = 'tmpTEST2/NIPSTEST2';
prefix = 'NIPSTEST2';

for dim = [10, 20]
    
    clear test
    
    W = [];
    for i = 1:nbWEIGHTS
        W = [W; randSimplex(dim)'];
    end
    
    clear i
    
    %%
    for k = 1:size(W,1)
        
        
        eReward = W(k,:);
        
        test(k).rew = eReward;
        
        clear results;
        
        i = 0;
        
        for nbEpisodes = [10, 50, 100, 500, 1000, 2000]
            
            i = i + 1;
            
            results(i).ep = nbEpisodes;
            results(i).seed = [];
            
            for kk = 1:length(gtypes)
                results(i).plane{kk} = [];
                results(i).plane_time{kk} = [];
            end
            
            for run = 1
                
                random_seed = randi([1000,2^30],1);
                results(i).seed = [results(i).seed;random_seed];
                
                delete([tmpFolder, '*']);
                
                strval = '';
                for oo = 1:length(eReward)
                    strval = [strval, ' ', num2str(eReward(oo))];
                end
                
                tic;
                status = system([cmd, ' ' num2str(random_seed) ' ' num2str(nbEpisodes), ...
                    ' ', num2str(length(eReward)), ' ', strval]);
                tcpp = toc;
                fprintf('Time GIRL C++: %f\n', tcpp);
                
                %% READ RESULTS
                for kk = 1:length(gtypes)
                    algorithm = gtypes{kk};
                    plane_R = dlmread([tmpFolder,'girl_plane_',algorithm,'.log']);
                    
                    A = dlmread([tmpFolder,'girl_time_',algorithm,'.log']);
                    plane_T = A(1,:);
                    clear A;
                    results(i).plane{kk} = [results(i).plane{kk};plane_R];
                    results(i).plane_time{kk} = [results(i).plane_time{kk};plane_T];
                end
                
                
                fprintf('\n -------------- \n\n');
                
            end
            
            sss = [tmpPrefix,'_D_',num2str(dim),'_W',num2str(k),'_ep',num2str(nbEpisodes),'.mat'];
            save(sss);
            
        end
        
        test(k).results = results;
    end
    
    sss = [prefix,'_D_',num2str(dim),'.mat'];
    save(sss);
end

%%
if 0
    clc;
    dimr = length(eReward);
    aa = repmat('c', 1, dimr);
    nalg = length(gtypes);
    ftab = ['{c', repmat(['|',aa],1, nalg), '}'];
    ftabtime = ['{c', repmat('|c',1, nalg*2), '}'];
    clear aa;
    clear i;
    for i = 1:length(test)
        %% PGIRL
        fprintf('\\begin{table}\n\\begin{small}\n\\centering\n\\begin{tabular}');
        fprintf('%s\n', ftab);
        fprintf('\\hline \n');
        %     fprintf('\\multirow{2}{*}{Episodes} & \\multicolumn{%d}{c|}{PGIRL} & \\multicolumn{%d}{c}{GIRL}\\\\ \n',nalg*dimr,nalg*dimr);
        fprintf('\\multirow{2}{*}{Episodes} & \\multicolumn{%d}{c|}{PGIRL} \\\\ \n',nalg*dimr);
        %     fprintf('& \\multicolumn{%d}{c|}{R} & \\multicolumn{%d}{c|}{RB}', dimr, dimr);
        %     fprintf('& \\multicolumn{%d}{c|}{G} & \\multicolumn{%d}{c|}{GB} ', dimr, dimr);
        fprintf('& \\multicolumn{%d}{c|}{R} & \\multicolumn{%d}{c|}{RB}', dimr, dimr);
        fprintf('& \\multicolumn{%d}{c|}{G} & \\multicolumn{%d}{c}{GB} \\\\ \n', dimr, dimr);
        fprintf('\\hline \n');
        for epel = 1:length(test(i).results)
            fprintf('\\multirow{2}{*}{$%d$} ', test(i).results(epel).ep);
            for alg = 1:length(test(i).results(epel).plane)
                clear valplane valgnorm
                valplane = test(i).results(epel).plane{alg};
                timeplane = test(i).results(epel).plane_time{alg};
                valgnorm = test(i).results(epel).gnorm{alg};
                timegnorm = test(i).results(epel).gnorm_time{alg};
                nruns = size(valplane,1);
                
                
                I = find(valplane < 0);
                if (size(I,1) > 0)
                    %                     valplane(valplane < 0) = 0;
                    %                     valplane = valplane ./ repmat(sum(valplane,2),1,size(valplane,2));
                    %                     valplane = abs(valplane);
                else
                    assert(max(max(abs(valplane-valgnorm)))<1e-2);
                end
                
                muplane = mean(valplane);
                mugnorm = mean(valgnorm);
                stdplane = std(valplane) / sqrt(nruns-1);
                stdgnorm = std(valgnorm) / sqrt(nruns-1);
                mutimeplane = mean(timeplane);
                mutimegnorm = mean(timegnorm);
                
                for nel = 1:size(valplane,2)
                    fprintf('& %.3f ', muplane(nel));
                end
                
                
                %             for nel = 1:size(valplane,2)
                %                 fprintf('& %.3f ', mugnorm(nel));
                %             end
            end
            fprintf('\\\\ \n');
            % stddev
            for alg = 1:length(test(i).results(epel).plane)
                valplane = test(i).results(epel).plane{alg};
                timeplane = test(i).results(epel).plane_time{alg};
                valgnorm = test(i).results(epel).gnorm{alg};
                timegnorm = test(i).results(epel).gnorm_time{alg};
                nruns = size(valplane,1);
                
                muplane = mean(valplane);
                mugnorm = mean(valgnorm);
                stdplane = std(valplane) / sqrt(nruns-1);
                stdgnorm = std(valgnorm) / sqrt(nruns-1);
                mutimeplane = mean(timeplane);
                mutimegnorm = mean(timegnorm);
                
                for nel = 1:size(valplane,2)
                    fprintf('& {\\scriptsize $\\pm %.3f$} ', stdplane(nel));
                end
                
                
                %             for nel = 1:size(valplane,2)
                %                 fprintf('& {\\scriptsize $\\pm %.3f$} ', stdgnorm(nel));
                %             end
            end
            fprintf('\\\\ \n');
            fprintf('\\hline \n');
        end
        fprintf('\\end{tabular} \n');
        strval = '';
        rew = test(i).rew;
        for oo = 1:dimr
            strval = [strval, ' ', num2str(rew(oo))];
        end
        fprintf('\\caption{Weights recovered starting form [%s]}',strval);
        fprintf('\\end{small}\n');
        fprintf('\\end{table}\n');
        
        %% GIRL
        fprintf('\\begin{table}\n\\begin{small}\n\\centering\n\\begin{tabular}');
        fprintf('%s\n', ftab);
        fprintf('\\hline \n');
        %     fprintf('\\multirow{2}{*}{Episodes} & \\multicolumn{%d}{c|}{PGIRL} & \\multicolumn{%d}{c}{GIRL}\\\\ \n',nalg*dimr,nalg*dimr);
        fprintf('\\multirow{2}{*}{Episodes} & \\multicolumn{%d}{c|}{GIRL} \\\\ \n',nalg*dimr);
        %     fprintf('& \\multicolumn{%d}{c|}{R} & \\multicolumn{%d}{c|}{RB}', dimr, dimr);
        %     fprintf('& \\multicolumn{%d}{c|}{G} & \\multicolumn{%d}{c|}{GB} ', dimr, dimr);
        fprintf('& \\multicolumn{%d}{c|}{R} & \\multicolumn{%d}{c|}{RB}', dimr, dimr);
        fprintf('& \\multicolumn{%d}{c|}{G} & \\multicolumn{%d}{c}{GB} \\\\ \n', dimr, dimr);
        fprintf('\\hline \n');
        for epel = 1:length(test(i).results)
            fprintf('\\multirow{2}{*}{$%d$} ', test(i).results(epel).ep);
            for alg = 1:length(test(i).results(epel).plane)
                valplane = test(i).results(epel).plane{alg};
                timeplane = test(i).results(epel).plane_time{alg};
                valgnorm = test(i).results(epel).gnorm{alg};
                timegnorm = test(i).results(epel).gnorm_time{alg};
                nruns = size(valplane,1);
                
                muplane = mean(valplane);
                mugnorm = mean(valgnorm);
                stdplane = std(valplane) / sqrt(nruns-1);
                stdgnorm = std(valgnorm) / sqrt(nruns-1);
                mutimeplane = mean(timeplane);
                mutimegnorm = mean(timegnorm);
                
                %             for nel = 1:size(valplane,2)
                %                 fprintf('& %.3f ', muplane(nel));
                %             end
                
                
                for nel = 1:size(valplane,2)
                    fprintf('& %.3f ', mugnorm(nel));
                end
            end
            fprintf('\\\\ \n');
            % stddev
            for alg = 1:length(test(i).results(epel).plane)
                valplane = test(i).results(epel).plane{alg};
                timeplane = test(i).results(epel).plane_time{alg};
                valgnorm = test(i).results(epel).gnorm{alg};
                timegnorm = test(i).results(epel).gnorm_time{alg};
                nruns = size(valplane,1);
                
                muplane = mean(valplane);
                mugnorm = mean(valgnorm);
                stdplane = std(valplane) / sqrt(nruns-1);
                stdgnorm = std(valgnorm) / sqrt(nruns-1);
                mutimeplane = mean(timeplane);
                mutimegnorm = mean(timegnorm);
                
                %             for nel = 1:size(valplane,2)
                %                 fprintf('& {\\scriptsize $\\pm %.3f$} ', stdplane(nel));
                %             end
                
                
                for nel = 1:size(valplane,2)
                    fprintf('& {\\scriptsize $\\pm %.3f$} ', stdgnorm(nel));
                end
            end
            fprintf('\\\\ \n');
            fprintf('\\hline \n');
        end
        fprintf('\\end{tabular} \n');
        strval = '';
        rew = test(i).rew;
        for oo = 1:dimr
            strval = [strval, ' ', num2str(rew(oo))];
        end
        fprintf('\\caption{Weights recovered starting form [%s]}',strval);
        fprintf('\\end{small}\n');
        fprintf('\\end{table}\n');
        
        %% TIME
        fprintf('\\begin{table}\n\\begin{small}\n\\centering\n\\begin{tabular}');
        fprintf('%s\n', ftabtime);
        fprintf('\\hline \n');
        fprintf('\\multirow{2}{*}{Episodes} & \\multicolumn{%d}{c|}{PGIRL} & \\multicolumn{%d}{c}{GIRL}\\\\ \n',nalg,nalg);
        fprintf('& R & RB & G & BG & R & RB & G & BG \\\\ \n');
        fprintf('\\hline \n');
        for epel = 1:length(test(i).results)
            fprintf('\\multirow{2}{*}{$%d$} ', test(i).results(epel).ep);
            for alg = 1:length(test(i).results(epel).plane)
                timeplane = test(i).results(epel).plane_time{alg};
                timegnorm = test(i).results(epel).gnorm_time{alg};
                nruns = size(timeplane,1);
                
                muplane = mean(timeplane);
                mugnorm = mean(timegnorm);
                stdplane = std(timeplane) / sqrt(nruns-1);
                stdgnorm = std(timegnorm) / sqrt(nruns-1);
                
                fprintf('& %.3f ', muplane);
                
                
                fprintf('& %.3f ', mugnorm);
                
            end
            fprintf('\\\\ \n');
            % stddev
            for alg = 1:length(test(i).results(epel).plane)
                timeplane = test(i).results(epel).plane_time{alg};
                timegnorm = test(i).results(epel).gnorm_time{alg};
                nruns = size(timeplane,1);
                
                muplane = mean(timeplane);
                mugnorm = mean(timegnorm);
                stdplane = std(timeplane) / sqrt(nruns-1);
                stdgnorm = std(timegnorm) / sqrt(nruns-1);
                
                fprintf('& {\\scriptsize $\\pm %.3f$} ', stdplane);
                
                fprintf('& {\\scriptsize $\\pm %.3f$} ', stdgnorm);
                
            end
            fprintf('\\\\ \n');
            fprintf('\\hline \n');
        end
        fprintf('\\end{tabular} \n');
        strval = '';
        rew = test(i).rew;
        for oo = 1:dimr
            strval = [strval, ' ', num2str(rew(oo))];
        end
        fprintf('\\caption{Times (s) [%s], non sono mediati perche'' salvavo solo l''ultimo (errore nello script)}',strval);
        fprintf('\\end{small}\n');
        fprintf('\\end{table}\n');
        fprintf('\\clearpage\n');
    end
end


%%
if 0
    for i = 1:length(gtypes)
        graph(i).data = [];
    end
    for i = 1:length(test)
        for epel = 1:length(test(i).results)
            for alg = 1:length(test(i).results(epel).plane)
                valgnorm = test(i).results(epel).gnorm{alg};
                valgnorm = valgnorm - repmat(test(i).rew,size(valgnorm,1),1);
                normv = zeros(size(valgnorm,1),1);
                for jj = 1:size(valgnorm,1)
                    normv(jj) = norm(valgnorm(jj,:),inf);
                end
                nruns = size(normv,1);
                mugnorm = mean(normv);
                stdgnorm = std(normv) / sqrt(nruns-1);
                graph(alg).data = [graph(alg).data; test(i).results(epel).ep mugnorm stdgnorm];
            end
        end
    end
    
    close all;
    figure(1);
    hold on;
    for i = 1:length(graph)
        % plot(graph(i).data(:,1), graph(i).data(:,2), 'o');
        errorbar(graph(i).data(:,1),graph(i).data(:,2),graph(i).data(:,3))
    end
    legend(gtypes{:})
    
    
    % %%
    % for i = 1:length(gtypes)
    %     graph(i).data = [];
    % end
    % for i = 1:length(test)
    %     for epel = 1:length(test(i).results)
    %         for alg = 1:length(test(i).results(epel).plane)
    %             valgnorm = test(i).results(epel).plane{alg};
    %                 I = find(valgnorm < 0);
    %                 if (size(I,1) > 0)
    %                     valgnorm(valgnorm < 0) = 0;
    %                     valgnorm = valgnorm ./ repmat(sum(valgnorm,2),1,size(valgnorm,2));
    %                     %                     valplane = abs(valplane);
    %                 end
    %
    %             valgnorm = valgnorm - repmat(test(i).rew,size(valgnorm,1),1);
    %             normv = zeros(size(valgnorm,1),1);
    %             for jj = 1:size(valgnorm,1)
    %                 normv(jj) = norm(valgnorm(jj,:),inf);
    %             end
    %             nruns = size(normv,1);
    %             mugnorm = mean(normv);
    %             stdgnorm = std(normv) / sqrt(nruns-1);
    %             graph(alg).data = [graph(alg).data; test(i).results(epel).ep mugnorm stdgnorm];
    %         end
    %     end
    % end
    %
    % % close all;
    % figure(2);
    % hold on;
    % for i = 1:length(graph)
    % % plot(graph(i).data(:,1), graph(i).data(:,2), 'o');
    % errorbar(graph(i).data(:,1),graph(i).data(:,2),graph(i).data(:,3))
    % end
    % legend(gtypes{:})
    
end
