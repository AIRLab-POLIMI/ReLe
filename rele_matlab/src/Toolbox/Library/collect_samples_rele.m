function [dataset, J] = collect_samples_rele( domain, maxepisodes, maxsteps, policy )
% Collect samples using ReLe toolbox: https://github.com/AIRLab-POLIMI/ReLe
% See README for details.

mdpconfig = [domain '_mdpvariables'];
mdp_var = feval(mdpconfig);
gamma = mdp_var.gamma;
n_obj = mdp_var.nvar_reward;
max_obj = mdp_var.max_obj;

mexParams.nbRewards = n_obj;

if strcmp('dam',domain)
    mexParams.penalize = mdp_var.penalize;
    if mexParams.penalize == 0
        mexParams.initType = 'random_discrete';
    else
        mexParams.initType = 'random';
    end
end

mexParams.policyParameters = policy.theta;
if nargout == 2
    [dataset, J] = collectSamples(domain, maxepisodes, maxsteps, gamma, mexParams);
else
    [~, J] = collectSamples(domain, maxepisodes, maxsteps, gamma, mexParams);
end
J = mean(J,1) .* abs(max_obj)';
