function [mexParams, gamma, max_obj] = getMexParams(domain)
% A wrapper to interface with ReLe Toolbox.

mdpconfig = [domain '_mdpvariables'];
mdp_var = feval(mdpconfig);
gamma = mdp_var.gamma;
n_obj = mdp_var.nvar_reward;
max_obj = mdp_var.max_obj;

mexParams.nbRewards = n_obj;
mexParams.penalize = 0; % water reservoir
mexParams.initType = 'random_discrete'; % water reservoir

end