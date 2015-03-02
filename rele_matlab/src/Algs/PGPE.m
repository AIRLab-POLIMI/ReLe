%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: Deisenroth, M. P.; Neumann, G.; Peters, J. (2013),
% A Survey on Policy Search for Robotics, Foundations and Trends
% in Robotics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
domain = 'deep';
robj = 1;
[~, pol_low, episodes, steps, gamma, avg_rew_setting, max_obj] = settings(domain);

% Parameters of the high-level policy
n_params = size(pol_low.theta,1);

% If the policy has a learnable variance, we don't want to learn it and
% we make it deterministic
n_params = n_params - pol_low.dim_variance_params;
pol_low = pol_low.makeDeterministic;

mu0 = zeros(n_params,1);
sigma0 = 10 * eye(n_params); % change according to the domain
tau = 50 * ones(size(diag(sigma0)));
pol_high = constant_logistic_gaussian_policy(n_params,mu0,diag(sigma0),tau);
% pol_high = constant_smart_gaussian_policy(n_params,mu0,sigma0);
% pol_high = constant_diag_gaussian_policy(n_params,mu0,sqrt(diag(sigma0)));

N = 150;
lrate = 1;

J = zeros(1,N);
J_history = [];
i = 0;

%% Learning
while true
    
    i = i + 1;
    
    Theta = zeros(n_params,N);
    num = 0;
    den = 0;
    dlogPidtheta = zeros(pol_high.dlogPidtheta,N);

    % Sampling
    parfor k = 1 : N
    
        % Extract theta from the high-level policy and perform a rollout
        pol_tmp = pol_low;
        theta = pol_high.drawAction;
        pol_tmp.theta(1:n_params) = theta;
        Theta(:,k) = theta;
        [ds, uJ, dJ] = collect_samples(domain,episodes,steps,pol_tmp,avg_rew_setting,gamma);
        if gamma == 1
            J(k) = uJ(robj) .* max_obj(robj);
        else
            J(k) = dJ(robj) .* max_obj(robj);
        end
        
        dlogPidtheta(:,k) = pol_high.dlogPidtheta(Theta(:,k));

        % Estimate optimal baseline
        num = num + dlogPidtheta(:,k).^2 * J(k);
        den = den + dlogPidtheta(:,k).^2;
        
    end
    
    mean_J = mean(J);
    fprintf( 'Iteration %d, Mean Reward: %f\n', i, mean_J );
    J_history = [J_history; mean_J];
	
    b = num ./ den;
    
    % Estimate gradient
    grad = 0;
    parfor k = 1 : N
        grad = grad + dlogPidtheta(:,k) .* (J(k) - b);
    end
    grad = grad / N;
    
    % Ending condition
    if norm(grad) < .001
        break
    end
    
    pol_high = pol_high.update( lrate * grad / max(norm(grad),1) );
    
end