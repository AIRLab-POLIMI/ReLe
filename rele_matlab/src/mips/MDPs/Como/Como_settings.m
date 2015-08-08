function [ n_obj, policy, episodes, steps, gamma ] = Como_settings
% Outputs:
% - n_obj    : number of objectives of the MDP
% - policy   : default policy
% - episodes : number of episodes for evaluation / learning
% - steps    : max number of steps per episode
% - gamma    : discount factor of the MDP

mdp_vars = Como_mdpvariables(); % gives out MDP variables according to Castelletti's article 
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;
dim = mdp_vars.nvar_action;

k0 = [1, 1, 1, 1, 1]; % starting value of weights
sigma0 = 10; % starting value for sigma

% k0 = zeros(1,dam_basis_rbf());
policy = gaussian_diag_linear(@Como_basis_rbf, dim, k0, sigma0); % This is the initial policy...

%%% Evaluation
episodes = 1000;
steps = 100;

%%% Learning
episodes = 50;
steps = 30;

end

