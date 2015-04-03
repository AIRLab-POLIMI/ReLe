%% NLS gradient test
addpath(genpath('../Statistics'));
addpath(genpath('./GradientTests'));
addpath('../');

%clear old data
clear all;
clc;

algorithms{1} = 'r';
algorithms{2} = 'g';
algorithms{3} = 'rb';
algorithms{4} = 'gb';

nbEpisodes = 50;
nbUpdates  = 1000;
stepLength = 0.001;

prog = '/home/matteo/Projects/github/ReLe/rele-build/nls_PG';
args = [num2str(nbUpdates), ' ', num2str(nbEpisodes), ...
    ' ', num2str(stepLength)];

figure(1);
hold on;
J = zeros(nbUpdates,length(algorithms));
for i = 1 : length(algorithms)
    cmd = [prog, ' ', algorithms{i}, ' ', args];
    status = system(cmd);
    
    %% show results
    disp('Reading agent data...')
    csv = csvread(['/tmp/ReLe/Nls/PG/Nls_',algorithms{i},'_agentData.log']);
    
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
    
    %     plot(J(:,i));
    shadedErrorBar(1:size(J_history,2), ...
        mean(J_history), ...
        2*sqrt(diag(cov(J_history))), ...
        {'LineWidth', 2'}, 1);
    legend(algorithms{i});
end
hold off;
figure(2);
hold on;
for i = 1:length(algorithms)
    plot(J(:,i))
end
legend(algorithms);
hold off;

%% Plot results
% figure; shadedErrorBar(1:size(J_history,2), ...
%     mean(J_history), ...
%     2*sqrt(diag(cov(J_history))), ...
%     {'LineWidth', 2'}, 1);
xlabel('Iterations')
ylabel('Average return')