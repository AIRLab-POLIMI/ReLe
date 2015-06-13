%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: M P Deisenroth, G Neumann, J Peters (2013)
% A Survey on Policy Search for Robotics, Foundations and Trends in
% Robotics
%
% With sample reuse.
% FMINCON solved iteratively for eta and theta separately.
%
% This is just an example in which the objectives represent the context.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Init domain and low-level policy
clear all

domain = 'deep';
[n_obj, pol_low, ~, steps] = settings(domain);

% If the low-level policy has a learnable variance, we don't want to learn 
% it and we make it deterministic
dim_theta = size(pol_low.theta,1) - pol_low.dim_variance_params;
pol_low = pol_low.makeDeterministic;

%% Init CREPS and high-level policy
phi_policy = @(varargin)basis_poly(1,n_obj,0,varargin{:});
phi_vfun = @(varargin)basis_poly(1,n_obj,0,varargin{:});

K0 = zeros(dim_theta,phi_policy());
mu0 = zeros(dim_theta,1);
Sigma0 = 100 * eye(dim_theta); % change according to the domain
% pol_high = gaussian_linear(phi_policy, dim_theta, W0, Sigma0);
% pol_high = gaussian_diag_linear(phi_policy, dim_theta, W0, sqrt(diag(Sigma0)));
pol_high = gaussian_linear_full(phi_policy, dim_theta, mu0, K0, Sigma0);

% In this example the context is defined by the weights of the objectives
possible_contexts = convexWeights(n_obj, 1);

epsilon = 0.9;
N = 50; % number of rollouts per iteration
N_MAX = N*10; % max number of rollouts used for the policy update
MAX_ROLLOUTS = 10000;
solver = CREPS_IW_Solver(epsilon,N,N_MAX,pol_high,phi_vfun);

J = zeros(1,N_MAX);
PhiVfun = zeros(N_MAX,solver.basis());
PhiPolicy = zeros(N_MAX,solver.policy.basis());
Theta = zeros(N_MAX,dim_theta);
Policies = pol_high.empty(N_MAX,0);

iter = 0;

%% Run CREPS
while iter < MAX_ROLLOUTS / N
    
    iter = iter + 1;

    context_samples = datasample(possible_contexts,N);
    Policies_iter(1:N) = solver.policy;
    
    PhiPolicy_iter = zeros(N,solver.policy.basis());
    PhiVfun_iter = zeros(N,solver.basis());
    Theta_iter = zeros(N,dim_theta);
    J_iter = zeros(1,N);

    % Sample theta
    parfor k = 1 : N
        % Get context
        context = context_samples(k,:)';
        PhiPolicy_iter(k,:) = solver.policy.basis(context);
        PhiVfun_iter(k,:) = solver.basis(context);
        
        % Draw theta from the Gaussian
        theta = solver.policy.drawAction(context);
        pol_tmp = pol_low;
        pol_tmp.theta(1:dim_theta) = theta; % set only the mean, not the variance
        Theta_iter(k,:) = theta;

        % Rollout
        [~, J_ep] = collect_samples(domain, 1, steps, pol_tmp);
        J_iter(k) = sum(J_ep .* context');
    end
    
    % At first run, fill the pool to maintain the samples distribution
    if iter == 1
        J(:) = min(J_iter);
        random_samples = datasample(possible_contexts,N_MAX);
        parfor k = 1 : N_MAX
            random_context = random_samples(k,:)'; % generate random context
            PhiPolicy(k,:) = solver.policy.basis(random_context);
            PhiVfun(k,:) = solver.basis(random_context);
            Theta(k,:) = solver.policy.drawAction(random_context);
        end
        Policies(1:N_MAX) = solver.policy;
    end
    
    % Enqueue the new samples and remove the old ones
    J = [J_iter, J(1:N_MAX-N)];
    Theta = [Theta_iter; Theta(1:N_MAX-N, :)];
    PhiVfun = [PhiVfun_iter; PhiVfun(1:N_MAX-N, :)];
    PhiPolicy = [PhiPolicy_iter; PhiPolicy(1:N_MAX-N, :)];
    Policies = [Policies_iter, Policies(1:N_MAX-N)];

    % Importance Sampling Weights
    W = zeros(1, N_MAX); % IS weights
    p = zeros(N_MAX, 1); % p(i) = probability of drawing Theta_i from policy p (p = target)
    Q = zeros(N_MAX, N_MAX); % Q(i,j) = probability of drawing Theta_i from policy q_j (q = sampling)
    alpha = ones(1, N_MAX) / N_MAX; % mixture responsibilities
    for i = 1 : N_MAX
        p(i) = solver.policy.evaluate(PhiPolicy(i,:)', Theta(i,:)');
        for j = 1 : N : N_MAX
            Q(i, j:j+N-1) = Policies(j).evaluate(PhiPolicy(i,:)', Theta(i,:)');
        end
%         W(i) = p(i) / Q(i,i); % Standard IW
        W(i) = p(i) / sum(alpha .* Q(i,:)); % Mixture IW
%         W(i) = p(i) / sum(Q(i,:)); % Daniel, IROS 2012
%         W(i) = 1;
    end
    
    % Get the weights for policy update
    [weights, divKL] = solver.optimize(J, PhiVfun, W);

    % Print a matrix [context, n_samples, mean_J]
    print_vector = zeros(size(possible_contexts,1), n_obj+2);
    for i = 1 : size(possible_contexts,1)
        indices = ismember(context_samples, possible_contexts(i,:), 'rows');
        res_i = J_iter(indices);
        print_vector(i,:) = [possible_contexts(i,:), length(res_i), mean(res_i)];
    end
    fprintf('\n%d) Context \t N. samples \t Mean Reward \n', iter)
    fprintf('[%.2f %.2f] \t\t %d \t\t %.4f\n', print_vector')
    fprintf( 'KL Divergence: %.4f, Entropy: %.3f \n', divKL, solver.policy.entropy );
    
    % Stopping condition
    if divKL < 1e-3
        break
    else
        solver.update(weights, Theta, PhiPolicy);
    end

end
