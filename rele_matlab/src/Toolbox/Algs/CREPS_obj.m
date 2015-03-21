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
[n_obj, pol_low, ~, steps, gamma, is_avg, max_obj] = settings(domain);

% If the low-level policy has a learnable variance, we don't want to learn 
% it and we make it deterministic
dim_theta = size(pol_low.theta,1) - pol_low.dim_variance_params;
pol_low = pol_low.makeDeterministic;

possible_contexts = [1 0; 0 1]; % in this example the objectives are the context

%% Init CREPS and high-level policy
phi_policy = @(varargin)basis_poly(1,n_obj,0,varargin{:});
phi_vfun = @(varargin)basis_poly(1,n_obj,0,varargin{:});

K0 = zeros(dim_theta,phi_policy());
mu0 = zeros(dim_theta,1);
Sigma0 = 100 * eye(dim_theta); % change according to the domain
% pol_high = smart_gaussian_policy(phi_policy, dim_theta, W0, Sigma0);
% pol_high = smart_diag_gaussian_policy(phi_policy, dim_theta, W0, sqrt(diag(Sigma0)));
pol_high = full_smart_gaussian_policy(phi_policy, dim_theta, mu0, K0, Sigma0);

epsilon = 0.9;
N = 25; % number of rollouts per iteration
N_MAX = 150; % max number of rollouts used for the policy update
MAX_ROLLOUTS = 5000;
N_eval = 0;
solver = CREPS_Solver(epsilon,N,N_eval,N_MAX,pol_high,phi_vfun);

J = zeros(1,N_MAX);
PhiVfun = zeros(N_MAX,solver.basis());
PhiPolicy = zeros(N_MAX,solver.policy.basis());
Theta = zeros(N_MAX,dim_theta);

iter = 0;

%% Run CREPS
while iter < MAX_ROLLOUTS / N
    
    iter = iter + 1;

    context_samples = datasample(possible_contexts,N);
    
    PhiPolicy_iter = zeros(N,solver.policy.basis());
    PhiVfun_iter = zeros(N,solver.basis());
    Theta_iter = zeros(N,dim_theta);
    J_iter = zeros(1,N);

    % Sample theta
    parfor k = 1 : N
        % Get context
        context = context_samples(k,:)';
        robj = find(context);
        PhiPolicy_iter(k,:) = solver.policy.basis(context);
        PhiVfun_iter(k,:) = solver.basis(context);
        
        % Draw theta from the Gaussian
        theta = solver.policy.drawAction(context);
        pol_tmp = pol_low;
        pol_tmp.theta(1:dim_theta) = theta; % set only the mean, not the variance
        Theta_iter(k,:) = theta;

        % Rollout
        [ds, uJ, dJ] = collect_samples(domain,1,steps,pol_tmp,is_avg,gamma);
        if gamma == 1
            J_iter(k) = uJ(robj) .* max_obj(robj);
        else
            J_iter(k) = dJ(robj) .* max_obj(robj);
        end
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
    end
    
    % Enqueue the new samples and remove the old ones
    J = [J_iter, J(1:N_MAX-N)];
    Theta = [Theta_iter; Theta(1:N_MAX-N, :)];
    PhiVfun = [PhiVfun_iter; PhiVfun(1:N_MAX-N, :)];
    PhiPolicy = [PhiPolicy_iter; PhiPolicy(1:N_MAX-N, :)];

    % Get the weights for policy update
    [weights, divKL] = solver.optimize(J, PhiVfun);

    % Print a matrix with [r_obj, n_samples per context, mean_J of context]
    print_vector = zeros(n_obj, 3);
    for i = 1 : n_obj
        indices = find(context_samples(:,i));
        res_i = J_iter(indices);
        print_vector(i,:) = [i, length(res_i), mean(res_i)];
    end
    fprintf('\n%d) Context \t N. samples \t Mean Reward \n', iter)
    fprintf('%.2f \t\t %d \t\t %.4f\n', print_vector')
    fprintf( 'KL Divergence: %.4f \n', divKL );
    
    % Stopping condition
    if divKL < 1e-3
        break
    else
        solver.update(weights, Theta, PhiPolicy);
    end

end
