function [ n_obj, policy, episodes, steps, gamma ] = unicycle_settings
% Outputs:
%
% - n_obj    : number of objectives of the MDP
% - policy   : default policy
% - episodes : number of episodes for evaluation / learning
% - steps    : max number of steps per episode
% - gamma    : discount factor of the MDP

mdp_vars = unicycle_mdpvariables();
n_obj    = mdp_vars.nvar_reward;
gamma    = mdp_vars.gamma;

k0 = zeros(3,1);
policy = unicycle_controllaw(k0);

%%% Evaluation
episodes = 1000;
steps = 300;

%%% Learning
episodes = 200;
steps = 300;

end