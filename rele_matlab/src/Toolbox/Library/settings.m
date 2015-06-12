function [ n_obj, policy, episodes, steps, gamma ] = settings( domain )
% Outputs:
%
% - n_obj    : number of objectives of the MDP
% - policy   : default policy
% - episodes : number of episodes for the evaluation
% - steps    : max number of steps per episode for the evaluation
% - gamma    : discount factor of the MDP

if strcmp('deep',domain)
    
    mdp_vars = deep_mdpvariables();
    policy = gibbs(@deep_basis_pol_v1, ...
        zeros(deep_basis_pol_v1,1), ...
        mdp_vars.action_list);
    episodes = 50;
    steps = 25;
    
elseif strcmp('mountain',domain)
    
    mdp_vars = mountain_mdpvariables();
    policy = gibbs(@mountain_basis_pol, ...
        zeros(mountain_basis_pol,1), ...
        mdp_vars.action_list);
    episodes = 100;
    steps = 10000;
    
elseif strcmp('puddle',domain)
    
    mdp_vars = puddle_mdpvariables();
    policy = gibbs(@puddle_basis_pol, ...
        zeros(puddle_basis_pol,1), ...
        mdp_vars.action_list);
    episodes = 2000;
    steps = 50;
    
elseif strcmp('resource',domain)
    
    mdp_vars = resource_mdpvariables();
    policy = gibbs(@resource_basis_pol, ...
        zeros(resource_basis_pol,1), ...
        mdp_vars.action_list);
    episodes = 150;
    steps = 40;
    
elseif strcmp('dam',domain)
    
    mdp_vars = dam_mdpvariables();
    dim = mdp_vars.nvar_action;
    k0 = [50, -50, 0, 0, 50];
    %     k0 = zeros(1,dam_basis_rbf());
    %     policy = gaussian_fixedvar(@dam_basis_rbf, dim, k0, 0.1);
    %     policy = gaussian_linear(@dam_basis_rbf, dim, k0, 20);
    policy = gaussian_diag_linear(@dam_basis_rbf, dim, k0, 20);
    %     policy = gaussian_logistic_linear(@dam_basis_rbf, dim, k0, 1, 50);
    episodes = 1000;
    steps = 100;
    
elseif strcmp('lqr',domain)
    
    mdp_vars = lqr_mdpvariables();
    dim = mdp_vars.nvar_action;
    k0 = -0.5 * eye(dim);
    s0 = 1 * eye(dim);
    offset0 = zeros(dim,1);
    tau = 1.3 * ones(size(diag(s0)));
    %     policy = gaussian_logistic_linear(@lqr_basis_pol, dim, k0, diag(s0), tau);
    %     policy = gaussian_linear(@lqr_basis_pol, dim, k0, s0);
    %     policy = gaussian_diag_linear(@lqr_basis_pol, dim, k0, diag(s0));
    %     policy = gaussian_linear_full(@lqr_basis_pol, dim, offset0, k0, (s0));
    %     policy = gaussian_chol_linear(@lqr_basis_pol, dim, k0, chol(s0));
    policy = gaussian_fixedvar(@lqr_basis_pol, dim, k0, s0);
    episodes = 150;
    steps = 50;
    
elseif strcmp('mce',domain)
    
    mdp_vars = mce_mdpvariables();
    policy = gibbs(@mce_basis_rbf_v1, ...
        zeros(mce_basis_rbf_v1,1), ...
        mdp_vars.action_list);
    episodes = 500;
    steps = 150;
    
elseif strcmp('heat',domain)
    
    mdp_vars = heat_mdpvariables();
    policy = gibbs(@heat_basis_rbf_v1, ...
        zeros(mce_basis_rbf_v1,1), ...
        mdp_vars.action_list);
    episodes = 200;
    steps = 200;
    
else
    
    error('Domain unknown.')
    
end

n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;

end
