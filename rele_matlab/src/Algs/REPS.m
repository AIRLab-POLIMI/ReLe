%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: Deisenroth, M. P.; Neumann, G.; Peters, J. (2013),
% A Survey on Policy Search for Robotics, Foundations and Trends
% in Robotics.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
domain = 'deep';
robj = 1;
[~, pol_low, episodes, steps, gamma, avg_rew_setting, max_obj] = settings(domain);

% Parameters of the upper-level policy
n_params = size(pol_low.theta,1);

% If the policy has a learnable variance, we don't want to learn it and
% we make it deterministic
n_params = n_params - pol_low.dim_variance_params;
pol_low = pol_low.makeDeterministic;

mu0 = zeros(n_params,1);
sigma0 = 100 * eye(n_params); % change according to the domain
tau = 50 * ones(size(diag(sigma0)));

% pol_high = constant_logistic_gaussian_policy(n_params,mu0,diag(sigma0),tau);
pol_high = constant_smart_gaussian_policy(n_params,mu0,sigma0);
% pol_high = constant_diag_gaussian_policy(n_params,mu0,sqrt(diag(sigma0)));

N = 50;
epsilon = 0.9;

J = zeros(1,N);
J_history = [];
iter = 0;

%% Learning
while true

    iter = iter + 1;
    
    Theta = zeros(n_params,N);
    % Sampling
    for k = 1 : N
        % Extract theta from the high-level policy and perform a rollout
        pol_tmp = pol_low;
        theta = pol_high.drawAction;
        pol_tmp.theta(1:n_params) = theta; % set only the mean, not the variance
        Theta(:,k) = theta;
        [ds, uJ, dJ] = collect_samples(domain,episodes,steps,pol_tmp,avg_rew_setting,gamma);
        if gamma == 1
            J(k) = uJ(robj) .* max_obj(robj);
        else
            J(k) = dJ(robj) .* max_obj(robj);
        end
    end
    
    mean_J = mean(J);
    J_history = [J_history; mean_J];
	
	% Solve the optimization problem and the find optimal eta
    options = optimset('GradObj', 'on', ...
        'Display', 'off', ...
        'MaxFunEvals', 300 * 5, ...
        'Algorithm', 'trust-region-reflective', ...
        'TolX', 10^-8, ...
        'TolFun', 10^-12, ...
        'MaxIter', 300);
    lowerBound = 1e-4; % eta > 0
    upperBound = 1e8; % instead of INF, to avoid numerical problems
    eta0 = 1;
    eta = fmincon(@(eta)REPS_dual_function(eta,J,epsilon), ...
        eta0, [], [], [], [], lowerBound, upperBound, [], options);

    % Perform weighted ML to update the high-level policy
    d = exp( (J - max(J)) / eta )';

    % Compute KL divergence
    qWeighting = ones(N,1);
    pWeighting = d;
    divKL = getKL(pWeighting, qWeighting);

    fprintf( 'Iteration %d, Mean Reward: %f, KL Divergence: %f\n', ...
        iter, mean_J, divKL );

    error = divKL - epsilon;
    if error > 0.1 * epsilon
        disp( 'WARNING! Error greater than 10%! ');
    end

    % stopping condition
    if divKL < 1e-3
        return
    end

    pol_high = pol_high.weightedMLUpdate(d,Theta);
    
end
