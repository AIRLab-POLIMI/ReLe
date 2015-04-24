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


for k = 1:size(W,1)
    
    
    eReward = W(k,:);
    
    test(k).rew = eReward;
    
    clear results;
    
    i = 0;
    
    for nbEpisodes = [10, 50,100, 500, 1000]
        
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
                results(i).plane_time{kk} = [results(i).time{kk};plane_T];
                results(i).gnorm{kk} = [results(i).gnorm{kk};gnorm_R];
                results(i).gnorm_time{kk} = [results(i).time{kk};gnorm_T];
            end
            
            
            fprintf('\n -------------- \n\n');
            
        end
        
    end
    
    test(k).results = results;
end

save test2.mat