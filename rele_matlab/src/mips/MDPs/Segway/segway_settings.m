function [ n_obj, policy, episodes, steps, gamma ] = segway_settings
% Outputs:
%
% - n_obj    : number of objectives of the MDP
% - policy   : default policy
% - episodes : number of episodes for evaluation / learning
% - steps    : max number of steps per episode
% - gamma    : discount factor of the MDP

mdp_vars = segway_mdpvariables();
n_obj    = mdp_vars.nvar_reward;
gamma    = mdp_vars.gamma;
dim      = mdp_vars.nvar_action;

k0 = zeros(segway_basis_rbf(),1);
% k0 = zeros(1,dam_basis_rbf());
% policy = gaussian_fixedvar(@segway_basis_rbf, dim, k0, 0.1);
% policy = gaussian_linear(@segway_basis_rbf, dim, k0, 20);
policy = gaussian_diag_linear(@segway_basis_rbf, dim, k0, 2);
% policy = gaussian_logistic_linear(@segway_basis_rbf, dim, k0, 1, 50);

%%% Evaluation
episodes = 1000;
steps = 300;

%%% Learning
episodes = 200;
steps = 300;

end