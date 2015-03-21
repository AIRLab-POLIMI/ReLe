clear all
domain = 'deep';
robj = 1;
[n_obj, pol_low] = settings(domain);

% If the policy has a learnable variance, we don't want to learn it and
% we make it deterministic (see 'collect_episodes')
n_params = size(pol_low.theta,1) - pol_low.dim_variance_params;

mu0 = zeros(n_params,1);
sigma0 = 1 * eye(n_params); % change according to the domain
tau = 50 * ones(size(diag(sigma0)));

% pol_high = constant_logistic_gaussian_policy(n_params,mu0,diag(sigma0),tau);
% pol_high = constant_smart_gaussian_policy(n_params,mu0,sigma0);
pol_high = constant_chol_gaussian_policy(n_params,mu0,chol(sigma0));
% pol_high = constant_diag_gaussian_policy(n_params,mu0,sqrt(diag(sigma0)));

N = 10;
N_MAX = 300;

% solver = REPS_Solver(0.9,N,N_MAX,pol_high);
solver = NES_Solver(.1,N,N_MAX,pol_high);

J = zeros(N_MAX,n_obj);
Theta = zeros(pol_high.dim,N_MAX);

J_history = [];
iter = 0;

%% Learning
while true

    iter = iter + 1;
    
    [J_iter, Theta_iter] = collect_episodes(domain, N, solver.policy);

    % At first run, fill the pool to maintain the samples distribution
    if iter == 1
        J = repmat(min(J_iter),N_MAX,1);
        for k = 1 : N_MAX
            Theta(:,k) = solver.policy.drawAction;
        end
    end
        
    % Enqueue the new samples and remove the old ones
    J = [J_iter; J(1:N_MAX-N,:)];
    Theta = [Theta_iter, Theta(:, 1:N_MAX-N)];

    % Perform an update step
    div = solver.step(J(:,robj), Theta);
    
    avgRew = mean(J_iter(:,robj));
    J_history = [J_history, J_iter(:,robj)];
    fprintf( 'Iter: %d, Avg Reward: %.4f, Div: %.2f\n', ...
        iter, avgRew, div );

    % Ending condition
    if div < 0.005
        break
    end
    
end

%% Plot results
figure; shadedErrorBar(1:size(J_history,2), ...
    mean(J_history), ...
    2*sqrt(diag(cov(J_history))), ...
    {'LineWidth', 2'}, 1);
xlabel('Iterations')
ylabel('Average return')