%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function grad = finite_difference(policy, n_samples, J_theta, domain, robj)

theta = policy.theta;
n_theta = size(theta,1);
var = .01 * eye(n_theta); % change accordingly to the problem
delta_theta = zeros(n_samples,n_theta);
J_perturb = zeros(n_samples,1);
% since finite-different methods have parameter based exploration,
% set the policy to be deterministic and use only 1 episode
[~, ~, episodes, steps, gamma, avg_rew_setting] = settings(domain); 
J_theta_ref = J_theta(robj) * ones(n_samples,1);

for i = 1 : steps
    theta_perturb = mvnrnd(theta,var);
    delta_theta(i,:) = theta_perturb - theta';
    p = policy;
    p.theta = theta_perturb;
    
    [~, uJ, dJ] = collect_samples(domain,episodes,steps,p,avg_rew_setting,gamma);
    if gamma == 1
        J_perturb(i) = uJ(robj);
    else
        J_perturb(i) = dJ(robj);
    end
end
delta_J = J_perturb - J_theta_ref;

lambda = .9; % Ridge regression factor
grad = ((delta_theta' * delta_theta + lambda * eye(n_theta)) \ delta_theta') * delta_J;

end
