%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: M P Deisenroth, G Neumann, J Peters (2013)
% A Survey on Policy Search for Robotics, Foundations and Trends
% in Robotics.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
domain = 'deep';
robj = 1;
[n_obj, pol_low] = settings(domain);

% If the policy has a learnable variance, we don't want to learn it and
% we make it deterministic (see 'collect_episodes')
dim_theta = size(pol_low.theta,1) - pol_low.dim_variance_params;

mu0 = zeros(dim_theta,1);
sigma0 = 100 * eye(dim_theta); % change according to the domain
tau = 50 * ones(size(diag(sigma0)));
% pol_high = constant_logistic_gaussian_policy(n_params,mu0,diag(sigma0),tau);
pol_high = constant_smart_gaussian_policy(dim_theta,mu0,sigma0);
% pol_high = constant_diag_gaussian_policy(n_params,mu0,sqrt(diag(sigma0)));

N = 20;
N_MAX = 5*N;
epsilon = 0.9;
solver = REPS_Solver(epsilon,N,N_MAX,pol_high);

J = zeros(N_MAX,n_obj);
Theta = zeros(dim_theta,N_MAX);

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
    
    [weights, divKL] = solver.optimize(J(:,robj));

    avgRew = mean(J_iter(:,robj));
    J_history = [J_history, J_iter(:,robj)];
    fprintf( 'Iter: %d, Avg Reward: %.4f, KL Div: %.2f\n', ...
        iter, avgRew, divKL );
    
    if divKL < 1e-2
        break
    else
        solver.update(weights, Theta);
    end
    
end
