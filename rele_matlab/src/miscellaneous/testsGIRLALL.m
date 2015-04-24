clear all; close all; clc;


gtypes{1} = 'r';
gtypes{2} = 'rb';
gtypes{3} = 'g';
gtypes{4} = 'gb';


W = [];
for i = 0:0.2:1
    for j = 0:0.2:1
        if (i+j <= 1)
            W = [W; i,j,1-i-j];
        end
    end
end
clear i j

%%
for k = 6:size(W,1)
    
    
    eReward = W(k,:);
    
    test(k).rew = eReward;
    
    clear results;
    
    i = 0;
    
    for nbEpisodes = [10, 50,100, 500]%, 1000]
        
        i = i + 1;
        
        results(i).ep = nbEpisodes;
        
        for kk = 1:length(gtypes)
            results(i).plane{kk} = [];
            results(i).gnorm{kk} = [];
            results(i).time{kk} = [];
        end
        
        for run = 1:3
            
            delete('/tmp/ReLe/lqr/GIRL/*');
            
            %             algorithm = 'rb';
            cmd = '/home/mpirotta/Projects/github/ReLe/rele-build/lqr_GIRLALL';
            
            gamma = 0.99; % remember to modify also C code
            strval = '';
            for oo = 1:length(eReward)
                strval = [strval, ' ', num2str(eReward(oo))];
            end
            
            tic;
            status = system([cmd, ' ' num2str(nbEpisodes), ...
                ' ', num2str(length(eReward)), ' ', strval]);
            tcpp = toc;
            fprintf('Time GIRL C++: %f\n', tcpp);
            
            %% READ RESULTS
            for kk = 1:length(gtypes)
                algorithm = gtypes{kk};
                plane_R = dlmread(['/tmp/ReLe/lqr/GIRL/girl_plane_',algorithm,'.log']);
                gnorm_R = dlmread(['/tmp/ReLe/lqr/GIRL/girl_gnorm_',algorithm,'.log']);
                
                A = dlmread(['/tmp/ReLe/lqr/GIRL/girl_time_',algorithm,'.log']);
                gnorm_T = A(1,:);
                plane_T = A(2,:);
                clear A;
                results(i).plane{kk} = [results(i).plane{kk};plane_R];
                results(i).plane_time{kk} = [results(i).plane_time{kk};plane_T];
                results(i).gnorm{kk} = [results(i).gnorm{kk};gnorm_R];
                results(i).gnorm_time{kk} = [results(i).gnorm_time{kk};gnorm_T];
            end
            
            
            fprintf('\n -------------- \n\n');
            
        end
        
    end
    
    test(k).results = results;
end

save test2.mat

%%
clc;
dimr = length(eReward);
aa = repmat('c', 1, dimr);
nalg = length(gtypes);
ftab = ['{c', repmat(['|',aa],1, nalg*2), '}'];
ftabtime = ['{c', repmat('|c',1, nalg*2), '}'];
clear aa;
clear i;
for i = 1:length(test)
    fprintf('\\begin{table}\n\\begin{small}\n\\centering\n\\begin{tabular}');
    fprintf('%s\n', ftab);
    fprintf('\\hline \n');
    fprintf('\\multirow{2}{*}{Episodes} & \\multicolumn{%d}{c|}{PGIRL} & \\multicolumn{%d}{c}{GIRL}\\\\ \n',nalg*dimr,nalg*dimr);
    fprintf('& \\multicolumn{%d}{c|}{R} & \\multicolumn{%d}{c|}{RB}', dimr, dimr);
    fprintf('& \\multicolumn{%d}{c|}{G} & \\multicolumn{%d}{c|}{GB} ', dimr, dimr);
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
            
            for nel = 1:size(valplane,2)
                fprintf('& %.3f ', muplane(nel));
            end
            
            
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
            
            for nel = 1:size(valplane,2)
                fprintf('& {\\scriptsize $\\pm %.3f$} ', stdplane(nel));
            end
            
            
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
        fprintf('\\multirow{1}{*}{$%d$} ', test(i).results(epel).ep);
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
%         % stddev
%         for alg = 1:length(test(i).results(epel).plane)
%             timeplane = test(i).results(epel).plane_time{alg};
%             timegnorm = test(i).results(epel).gnorm_time{alg};
%             nruns = size(timeplane,1);
%             
%             muplane = mean(timeplane);
%             mugnorm = mean(timegnorm);
%             stdplane = std(timeplane) / sqrt(nruns-1);
%             stdgnorm = std(timegnorm) / sqrt(nruns-1);
%             
%                 fprintf('& {\\scriptsize $\\pm %.3f$} ', stdplane);
%             
%                 fprintf('& {\\scriptsize $\\pm %.3f$} ', stdgnorm);
%             
%         end
%         fprintf('\\\\ \n');
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