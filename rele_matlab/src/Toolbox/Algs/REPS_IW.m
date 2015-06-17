%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: M P Deisenroth, G Neumann, J Peters (2013)
% A Survey on Policy Search for Robotics, Foundations and Trends
% in Robotics.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
domain = 'deep';
robj = 1;
[n_obj, n_params, mu0, sigma0] = settings_episodic(domain,1);

pol_high = gaussian_constant(n_params,mu0,sigma0);
% pol_high = gaussian_diag_constant(n_params,mu0,sqrt(diag(sigma0)));

N = 20;
N_MAX = 10*N;
epsilon = 0.5;
solver = REPS_IW_Solver(epsilon,N,N_MAX,pol_high);

J = zeros(N_MAX,n_obj);
Theta = zeros(n_params,N_MAX);
Policies = pol_high.empty(N_MAX,0);

J_history = [];
iter = 0;

%% Learning
while true
    
    iter = iter + 1;
    
    [J_iter, Theta_iter] = collect_episodes(domain, N, solver.policy);
    Policies_iter(1:N) = solver.policy;

    % At first run, fill the pool to maintain the samples distribution
    if iter == 1
        J = repmat(min(J_iter),N_MAX,1);
        for k = 1 : N_MAX
            Theta(:,k) = solver.policy.drawAction;
            Policies(k) = solver.policy;
        end
    end
    
    % Enqueue the new samples and remove the old ones
    J = [J_iter; J(1:N_MAX-N,:)];
    Theta = [Theta_iter, Theta(:, 1:N_MAX-N)];
    Policies = [Policies_iter, Policies(1:N_MAX-N)];

    % Importance Sampling Weights
    W = zeros(N_MAX, 1); % IS weights
    p = zeros(N_MAX, 1); % p(i) = probability of drawing Theta_i from policy p (p = target)
    Q = zeros(N_MAX, N_MAX); % Q(i,j) = probability of drawing Theta_i from policy q_j (q = sampling)
    alpha = ones(1, N_MAX) / N_MAX; % mixture responsibilities
    for i = 1 : N_MAX
        p(i) = solver.policy.evaluate(Theta(:,i));
        for j = 1 : N : N_MAX
            Q(i, j:j+N-1) = Policies(j).evaluate(Theta(:,i));
        end
%         W(i) = p(i) / Q(i,i); % Standard IW
        W(i) = p(i) / sum(alpha .* Q(i,:)); % Mixture IW
%         W(i) = p(i) / sum(Q(i,:)); % Daniel, IROS 2012
%         W(i) = 1;
    end
    [weights, divKL] = solver.optimize(J(:,robj),W);

    avgRew = mean(J_iter(:,robj));
    J_history = [J_history, J_iter(:,robj)];
    fprintf( 'Iter: %d, Avg Reward: %.4f, KL Div: %.2f, Entropy: %.3f\n', ...
        iter, avgRew, divKL, solver.policy.entropy );
    
    if divKL < 1e-2
        break
    else
        solver.update(weights, Theta);
    end
    
end
