%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function grad = finite_difference(policy, n_samples, J_theta, domain, robj)

theta = policy.theta;
n_theta = size(theta,1);
var = 1 * eye(n_theta); % change according to the problem
delta_theta = zeros(n_samples,n_theta);
J_perturb = zeros(n_samples,1);
[~, ~, ~, steps, gamma, is_avg, max_obj] = settings(domain); 
J_theta_ref = J_theta(robj) * ones(n_samples,1);

for i = 1 : n_samples
    theta_perturb = mvnrnd(theta,var);
    delta_theta(i,:) = theta_perturb - theta';
    p = policy;
    p.theta = theta_perturb;
    
    [~, uJ, dJ] = collect_samples(domain,1,steps,p,is_avg,gamma);
    if gamma == 1
        J_perturb(i) = uJ(robj) .* max_obj(robj);
    else
        J_perturb(i) = dJ(robj) .* max_obj(robj);
    end
end
delta_J = J_perturb - J_theta_ref;

lambda = .9; % Ridge regression factor
grad = ((delta_theta' * delta_theta + lambda * eye(n_theta)) \ delta_theta') * delta_J;

end