%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: Deisenroth, M. P.; Neumann, G.; Peters, J. (2013),
% A Survey on Policy Search for Robotics, Foundations and Trends in
% Robotics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Change the basis functions if you wish. Also, you can use different bases
% for the upper-level policy and for the value function.

% In this implementation the contexts are the goals.

clear all
domain = 'deep';
[n_obj, pol_low, episodes, steps, gamma, avg_rew_setting, max_obj] = settings(domain);

% Parameters of the (Gaussian) distribution (upper-level policy) that generates parameters theta
n_params = size(pol_low.theta,1);

% If the low-level policy has a learnable variance, we don't want to learn 
% it and we make it deterministic
n_params = n_params - pol_low.dim_variance_params;
pol_low = pol_low.makeDeterministic;

dim_basis_policy = CREPS_basis;
dim_basis_vfun = CREPS_basis;
W0 = zeros(n_params,dim_basis_policy);
mu0 = zeros(n_params,1);
Sigma0 = 100 * eye(n_params); % change according to the domain
% pol_high = smart_gaussian_policy(@CREPS_basis, n_params, W0, Sigma0);
% pol_high = smart_diag_gaussian_policy(@CREPS_basis, n_params, W0, sqrt(diag(Sigma0)));
pol_high = full_smart_gaussian_policy(@CREPS_basis, n_params, mu0, W0, Sigma0);

N = 10;
N_MAX = 50;
epsilon = 0.9;

% Sample pool
J = zeros(1,N_MAX);
PhiVfun = zeros(N_MAX,dim_basis_vfun);
PhiPolicy = zeros(N_MAX,dim_basis_policy);
Theta = zeros(N_MAX,n_params);

iter = 0;

%% Learning
while true

    % Cells used to save J for each context during sampling
    results = cell(n_obj,1);
    iter = iter + 1;
    
    PhiPolicy_iter = zeros(N,dim_basis_policy);
    PhiVfun_iter = zeros(N,dim_basis_vfun);
    Theta_iter = zeros(N,n_params);
    J_iter = zeros(1,N);

    % Sample theta
    for k = 1 : N % to use the parfor you need to not save 'results{.}'
        % Generate context
        robj = randi(n_obj);
        context = [robj == 1 : n_obj]';
        PhiPolicy_iter(k,:) = CREPS_basis(context);
        PhiVfun_iter(k,:) = CREPS_basis(context); % use different bases if you want
        
        % Extract theta from the Gaussian
        theta = pol_high.drawAction(context);
        pol_low.theta(1:n_params) = theta; % set only the mean, not the variance
        Theta_iter(k,:) = theta;

        % Collect samples with one theta
        [ds, uJ, dJ] = collect_samples(domain,episodes,steps,pol_low,avg_rew_setting,gamma);
        if gamma == 1
            J_iter(k) = uJ(robj) .* max_obj(robj);
        else
            J_iter(k) = dJ(robj) .* max_obj(robj);
        end
        results{robj} = [results{robj}, J_iter(k)];
    end

    % At first run fill the pool to maintain the samples distribution
    if iter == 1
        J(:) = min(J_iter);
        for k = 1 : N_MAX
            r = rand;
            random_context = [r > 0.5, r <= 0.5]'; % generate random context
            PhiPolicy(k,:) = CREPS_basis(random_context);
            PhiVfun(k,:) = CREPS_basis(random_context);
            random_theta = pol_high.drawAction(random_context);
            Theta(k,:) = random_theta;
        end
    end
    
    % Enqueue the new samples and remove the old ones
    J = [J_iter, J(1:N_MAX-N)];
    Theta = [Theta_iter; Theta(1:N_MAX-N, :)];
    PhiVfun = [PhiVfun_iter; PhiVfun(1:N_MAX-N, :)];
    PhiPolicy = [PhiPolicy_iter; PhiPolicy(1:N_MAX-N, :)];
    
    % Solve the optimization problem
    options = optimset('GradObj', 'on', 'Display', 'off', ...
        'MaxFunEvals', 300 * 5, 'Algorithm', 'trust-region-reflective', ...
        'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 300);
    lowerBound = [-ones(dim_basis_vfun, 1) * 1e8; 1e-4];
    upperBound = [ones(dim_basis_vfun, 1) * 1e8; 1e8];
    x0 = ones(dim_basis_vfun+1,1);
    params = fmincon(@(params)CREPS_duals.full(params,J,epsilon,PhiVfun), ...
        x0, [], [], [], [], lowerBound, upperBound, [], options);
    v = params(1:end-1);
    eta = params(end);
    
    % Numerical trick
    advantage = J - v' * PhiVfun';
    maxAdvantage = max(advantage);

    % Perform weighted ML to update high-level policy parameters
    d = exp( (advantage - maxAdvantage) / eta )';
    
    % Compute KL divergence
    qWeighting = ones(N,1);
    pWeighting = d;
    divKL = getKL(pWeighting, qWeighting);

    % Print a matrix with [r_obj, n_samples per context, mean_J of context]
    print_vector = zeros(n_obj, 3);
    for i = 1 : n_obj
        print_vector(i,:) = [i, numel(results{i}), mean(results{i})];
    end
    fprintf('\n%g) Context \t N. samples \t Mean Reward \n', iter)
    fprintf('%f \t %d \t\t %f\n', print_vector')
    fprintf( 'KL Divergence: %f ', divKL );

    error = divKL - epsilon;
    if error > 0.1 * epsilon
        fprintf( '- WARNING! Error greater than 10%%! ');
    end
    fprintf('\n')

    % Stopping condition
    if divKL < 1e-3
        return
    end
    
    pol_high = pol_high.weightedMLUpdate(d,Theta,PhiPolicy);
    
end
