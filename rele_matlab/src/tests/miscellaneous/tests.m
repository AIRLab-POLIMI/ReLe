clear all; close all; clc;

gtypes{1} = 'rb';
gtypes{2} = 'gb';
gtypes{3} = 'g';
gtypes{4} = 'r';

W = [];
for i = 0:0.2:1
    for j = 0:0.2:1
        if (i+j <= 1)
            W = [W; i,j,1-i-j];
        end
    end
end
clear i j

for u = 1:length(gtypes)
    algorithm = gtypes{u};
    
    clear test;
    
    for k = 1:size(W,1)
        
        
        eReward = W(k,:);
        
        test(k).rew = eReward;
        
        clear results;
        
        i = 0;
        
        for nbEpisodes = [10, 50,100, 500, 1000]
            
            i = i + 1;
            
            results(i).ep = nbEpisodes;
            results(i).plane = [];
            results(i).gnorm = [];
            results(i).time = [];
            
            for run = 1:3
                
                delete('/tmp/ReLe/lqr/GIRL/*');
                
                %             algorithm = 'rb';
                cmd = '/home/mpirotta/Projects/github/ReLe/rele-build/lqr_GIRL';
                
                gamma = 0.99; % remember to modify also C code
                strval = '';
                for oo = 1:length(eReward)
                    strval = [strval, ' ', num2str(eReward(oo))];
                end
                
                tic;
                status = system([cmd, ' ', algorithm, ' ' num2str(nbEpisodes), ...
                    ' ', num2str(length(eReward)), ' ', strval]);
                tcpp = toc;
                fprintf('Time GIRL C++: %f\n', tcpp);
                
                %% READ RESULTS
                plane_R = dlmread(['/tmp/ReLe/lqr/GIRL/girl_plane_',algorithm,'.log']);
                gnorm_R = dlmread(['/tmp/ReLe/lqr/GIRL/girl_gnorm_',algorithm,'.log']);
                
                
                A = dlmread(['/tmp/ReLe/lqr/GIRL/timer.log']);
                gnorm_T = A(1,:);
                plane_T = A(2,:);
                clear A;
                
                dim = size(gnorm_R,2);
                
                fprintf('\n');
                disp(' Plane weights (C++):');
                disp(plane_R);
                disp(' Gnorm weights (C++):');
                disp(gnorm_R);
                fprintf('gnorm time: %f\n', gnorm_T);
                fprintf('plane time: %f\n', plane_T);
                fprintf('--------\n');
                
                
                results(i).plane = [results(i).plane;plane_R];
                results(i).plane_time = [results(i).time;plane_T];
                results(i).gnorm = [results(i).gnorm;gnorm_R];
                results(i).gnorm_time = [results(i).time;gnorm_T];
            end
            
        end
        
        test(k).results = results;
    end
    
    globalTest.test = test;
    globalTest.alg = algorithm;
    
end
save test2.mat