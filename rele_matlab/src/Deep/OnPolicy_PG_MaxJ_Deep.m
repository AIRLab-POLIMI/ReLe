%% DEEP gradient test
addpath(genpath('../Statistics'));
addpath('../');

%clear old data
clear all;
clc;

% algorithms{1} = 'r';
% algorithms{2} = 'g';
% algorithms{3} = 'rb';
% algorithms{4} = 'gb';
% algorithms{5} = 'gsb';
% algorithms{6} = 'natg';
% algorithms{7} = 'natr';
% algorithms{8} = 'enac';
 algorithms{1} = 'gb';

nbEpisodes = 50;
nbUpdates  = 1000;
stepLength = 0.001;

domain = 'deep';

path = '/home/dave/ReLe/devel/lib/rele/';
%path = '/home/matteo/Projects/github/ReLe/rele-build/';
prog = [path,domain,'_PG'];


% figure(1);
% hold on;
J = zeros(nbUpdates,length(algorithms));
for i = 1 : length(algorithms)
    
%    if strcmp(algorithms{i}, 'ng')
%        args = [num2str(nbUpdates), ' ', num2str(nbEpisodes), ...
%            ' ', '0.1'];
%    elseif strcmp(algorithms{i}, 'enac')
%        args = [num2str(nbUpdates), ' ', num2str(nbEpisodes), ...
%            ' ', '11000'];
%    else
        args = [num2str(nbUpdates), ' ', num2str(nbEpisodes), ...
            ' ', num2str(stepLength)];
%    end
    
    cmd = [prog, ' ', algorithms{i}, ' ', args];
    status = system(cmd);
    
    %% show results
    disp('Reading agent data...')
    csv = csvread(['/tmp/ReLe/',domain,'/PG/',domain,'_',algorithms{i},'_agentData.log']);
    
    disp('Organizing data...')
    
    index = 1;
    ep = 1;
    
    while(index < size(csv, 1))
        [data(ep), index] = ReadGradientStatistics(csv, index);
        ep = ep + 1;
    end
    
    clearvars csv
    
    
    J_history = [];
    for k = 1:length(data)
        J(k,i) = mean(data(k).J);
        J_history = [J_history, data(k).J'];
    end
    
%     %     plot(J(:,i));
%     shadedErrorBar(1:size(J_history,2), ...
%         mean(J_history), ...
%         2*sqrt(diag(cov(J_history))), ...
%         {'LineWidth', 2'}, 1);
%     legend(algorithms{i});
end
% hold off;
%%
figure(2);
hold on;
for i = 1:length(algorithms)
    plot(smooth(J(:,i)), 'Linewidth', 1.5)
%     plot(J(:,i), 'Linewidth', 1.5)
%     disp(algorithms{i})
%     pause
end
grid on;
title(['Deep, episodes:' num2str(nbEpisodes),', iter:',num2str(nbUpdates)]);
legend(algorithms, 'location', 'southeast');
hold off;

%% Plot results
% figure; shadedErrorBar(1:size(J_history,2), ...
%     mean(J_history), ...
%     2*sqrt(diag(cov(J_history))), ...
%     {'LineWidth', 2'}, 1);
xlabel('Iterations')
ylabel('Average return')
