function [ n_obj, policy, episodes, steps, gamma ] = nls_settings
% Outputs:
%
% - n_obj    : number of objectives of the MDP
% - policy   : default policy
% - episodes : number of episodes for evaluation / learning
% - steps    : max number of steps per episode
% - gamma    : discount factor of the MDP

mdp_vars = nls_mdpvariables();
n_obj    = mdp_vars.nvar_reward;
gamma    = mdp_vars.gamma;
dim      = mdp_vars.nvar_action;

k0 = [-0.4; 0.4];
policy = gaussian_statedepstddev_linear(@nls_basis_v1, dim, k0, @nls_cov_v1);

%%% Evaluation
episodes = 1000;
steps = 80;

%%% Learning
episodes = 200;
steps = 80;

end

