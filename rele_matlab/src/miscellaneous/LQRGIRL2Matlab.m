addpath(genpath('~/Dropbox/IRL_matlab'));
clear all; close all; clc;

TESTMATLAB = 0;

delete('/tmp/ReLe/lqr/GIRL/*');
algorithm = 'rb';
cmd = '/home/mpirotta/Projects/github/ReLe/rele-build/lqr_GIRL';

nbEpisodes = 200;
gamma = 0.99; % remember to modify also C code
eReward = [0.08; 0.67; 0.25];
strval = '';
for i = 1:length(eReward)
    strval = [strval, ' ', num2str(eReward(i))];
end

tic;
status = system([cmd, ' ', algorithm, ' ' num2str(nbEpisodes), ...
    ' ', num2str(length(eReward)), ' ', strval]);
tcpp = toc;
fprintf('Time GIRL C++: %f\n', tcpp);

%% READ RESULTS
plane_R = dlmread(['/tmp/ReLe/lqr/GIRL/girl_plane_',algorithm,'.log']);
gnorm_R = dlmread(['/tmp/ReLe/lqr/GIRL/girl_gnorm_',algorithm,'.log']);

dim = size(gnorm_R,2);

fprintf('\n');
disp(' Plane weights (C++):');
disp(plane_R);
disp(' Gnorm weights (C++):');
disp(gnorm_R);

if TESTMATLAB == 1
    %% READ DATASET
    disp('Reading data trajectories...')
    csv = csvread('/tmp/ReLe/lqr/GIRL/data.log');
    
    disp('Organizing data in episodes...')
    episodes = readDataset(csv);
    clearvars csv
    
    clear dataset
    for i = 1:length(episodes)
        dataset(i).s     = episodes(i).x(1:end-1,:)';
        dataset(i).a     = episodes(i).u(1:end-1,:)';
        dataset(i).nexts = episodes(i).x(2:end,:)';
        dataset(i).r     = episodes(i).r(1:end-1,:)';
    end
    nbEpisodes = length(dataset);
    
    %% REPLICATE RESULTS
    domain = 'lqr';
    mdpconfig  = irl_lqr_init ( dim );
    mdpconfig.gamma = gamma;
    tic;
    K = irl_lqr_calcolaPoliticaOttimaWS(mdpconfig, eReward);
    t = toc;
    fprintf('Optimal policy computation: %f\n', t);
    
    [n_obj, pol, ~, ~, ~, is_avg, ~] = settings(domain);
    assert(n_obj == dim);
    assert(length(pol.theta) == length(K(:)));
    init_pol = pol;
    init_pol.theta = K(:);
    clear pol;
    
    assert(init_pol.dlogPidtheta == dim);
    
    if strcmp(domain,'lqr')
        rewards{1} = @(s,a,ns) irl_lqr_reward(s,a,ns,mdpconfig,1);
        rewards{2} = @(s,a,ns) irl_lqr_reward(s,a,ns,mdpconfig,2);
        if dim >= 3
            rewards{3} = @(s,a,ns) irl_lqr_reward(s,a,ns,mdpconfig,3);
        end
        if dim >= 4
            rewards{4} = @(s,a,ns) irl_lqr_reward(s,a,ns,mdpconfig,4);
        end
        if dim >= 5
            rewards{5} = @(s,a,ns) irl_lqr_reward(s,a,ns,mdpconfig,5);
        end
        rewardf  = @(s,a,ns,w) irl_lqr_reward_ws(s,a,ns,w,mdpconfig);
        drewardf = @(s,a,ns,w) irl_lqr_reward_ws_diff(s,a,ns,w,mdpconfig);
    else
        error('unknown domain');
    end
    
    % Plane method
    tic;
    [w, gradients] = IRLgrad_plane(init_pol, rewards, dataset, gamma, algorithm);
    tp = toc;
    fprintf('Plane time: %f\n', tp);
    
    gplane.val  = w;
    gplane.grad = gradients;
    gplane.alg  = algorithm;
    gplane.time = tp;
    
    gplane.val'
    
    % gradient norm
    objfun = @(x) IRLgrad_step_objfun(x, rewardf, drewardf, ...
        init_pol, dataset, gamma, algorithm);
    
    %     'Algorithm', 'interior-point', ...
    %         'Display', 'off', ...
    options = optimset('GradObj', 'on', ...
        'Display', 'off', ...
        'MaxFunEvals', 300 * 5, ...
        'TolX', 10^-12, ...
        'TolFun', 10^-12, ...
        'MaxIter', 300);
    x0 = ones(dim,1)/dim;
    tic;
    [x,fval,exitflag,output] = fmincon(objfun, x0, ...
        -eye(dim), zeros(dim,1), ones(1,dim), 1,[], [], [], options);
    tg = toc;
    fprintf('Gnorm time: %f\n', tg);
    
    gnorm.val  = x;
    gnorm.fval = fval;
    gnorm.out  = output;
    gnorm.alg  = algorithm;
    gnorm.time = tg;
    
    gnorm.val'
    
    
    %% check
    fprintf('\n[ep %d] Real reward: ', nbEpisodes);
    fprintf('%f ', eReward/sum(eReward));
    fprintf('\n\n');
    for i = 1:length(gplane)
        fprintf('%4s | ', gplane(i).alg);
        fprintf('PLANE: ');
        fprintf('%9.6f ', gplane(i).val');
        fprintf('(t: %f s) | ', gplane(i).time);
        fprintf('GNORM: ');
        fprintf('%9.6f ', gnorm(i).val');
        fprintf('(t: %f s) ', gnorm(i).time);
        %                 fprintf(', inf-norm error: %f', norm(eReward/sum(eReward)-outputs(i).val', inf));
        fprintf('\n');
    end
    
    % %%
    % cgrad = dlmread('/tmp/ReLe/lqr/GIRL/grad.log');
    % cgrad,gradients
    
    assert(max(abs(gnorm.val'- gnorm_R))<=1e-5);
    assert(max(abs(gplane.val'- plane_R))<=1e-5);
end