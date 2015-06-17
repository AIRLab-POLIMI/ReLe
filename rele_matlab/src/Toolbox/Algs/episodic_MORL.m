% Weighted sum with episodic RL algorithms for multi-objective problems.

clear all
domain = 'dam';
verboseOut = 1;
makeDet = 0; % evaluate only deterministic policies?
[n_obj, n_params, mu0, sigma0] = settings_episodic(domain,1);

% pol_high = gaussian_constant(n_params,mu0,sigma0);
% pol_high = gaussian_chol_constant(n_params,mu0,chol(sigma0));
pol_high = gaussian_diag_constant(n_params,mu0,sqrt(diag(sigma0)));

N = 25;
N_MAX = N;

J = zeros(N_MAX,n_obj);
Theta = zeros(pol_high.dim,N_MAX);

step = 20; % density of the weights
W = convexWeights(n_obj, step);
n_sol = size(W,1);
front_J = zeros(n_sol, n_obj); % Pareto-frontier solutions
front_pol = cell(n_sol,1);
iter = 0;
maxLevel = 15;


%% Learning
parfor i = 1 : size(W,1)
    
    J = zeros(N_MAX,n_obj);
    Theta = zeros(pol_high.dim,N_MAX);
    solver = REPS_Solver(0.9,N,N_MAX,pol_high);
    % solver = NES_Solver(1,N,N_MAX,pol_high);
    
    level = 1;
    str_dir = strtrim(sprintf('%.3f, ', W(i,:)));
    str_dir(end) = [];

    %% Single objective optimization
    while true
        
        iter = iter + 1;
        
        [J_iter, Theta_iter] = collect_episodes(domain, N, solver.policy);
        
        % At first run, fill the pool to maintain the samples distribution
        if level == 1
            J = repmat(min(J_iter),N_MAX,1);
            for k = 1 : N_MAX
                Theta(:,k) = solver.policy.drawAction;
            end
        end
        
        % Enqueue the new samples and remove the old ones
        J = [J_iter; J(1:N_MAX-N,:)];
        Theta = [Theta_iter, Theta(:, 1:N_MAX-N)];
        
        % Perform an update step
        div = solver.step(sum(bsxfun(@times, W(i,:), J),2), Theta);
        
        str_obj = strtrim(sprintf('%.4f, ', mean(J)));
        str_obj(end) = [];
        if verboseOut, fprintf('[ %s ] || LV %d ) Dev: %.4f, \t J = [ %s ] \n', ...
                str_dir, level, div, str_obj); end

        % Ending condition
        if div < 0.005 || level >= maxLevel
            break
        end
        
        level = level + 1;
        
    end
    
    front_J(i,:) = mean(J);
    front_pol{i} = solver.policy;
    
end

front_pol = vertcat(front_pol{:});


%% Plot
fr = evaluate_policies_episodic(front_pol, domain, makeDet);
[f, p] = pareto(fr, front_pol);

figure; hold all
if n_obj == 2
    plot(f(:,1),f(:,2),'g+')
end

if n_obj == 3
    scatter3(f(:,1),f(:,2),f(:,3),'g+')
end

getReferenceFront(domain,1);
