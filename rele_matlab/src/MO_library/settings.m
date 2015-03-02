function [ n_obj, policy, episodes, steps, gamma, avg_rew_setting, max_obj ] ...
    = settings( domain )

if strcmp('deep',domain)

    mdp_vars = deep_mdpvariables();
    policy = gibbs_policy(@deep_basis_pol_v1, ...
        zeros(deep_basis_pol_v1,1), ...
        mdp_vars.action_list);
    episodes = 1;
    steps = 50;
    gamma = 1;
    avg_rew_setting = 0;
    
elseif strcmp('mountain',domain)

    mdp_vars = mountain_mdpvariables();
    policy = gibbs_policy(@mountain_basis_pol, ...
        zeros(mountain_basis_pol,1), ...
        mdp_vars.action_list);
    episodes = 100;
    steps = 10000;
    gamma = 1;
    avg_rew_setting = 0;

elseif strcmp('puddleworld',domain)

    mdp_vars = puddleworld_mdpvariables();
    policy = gibbs_policy(@puddleworld_basis_pol, ...
        zeros(puddleworld_basis_pol,1), ...
        mdp_vars.action_list);
    episodes = 2000;
    steps = 50;
    gamma = 1;
    avg_rew_setting = 0;
    
elseif strcmp('resource',domain)

    mdp_vars = resource_mdpvariables();
    policy = gibbs_policy(@resource_basis_pol, ...
        zeros(resource_basis_pol,1), ...
        mdp_vars.action_list);
    episodes = 150;
    steps = 40;
    gamma = 1;
    avg_rew_setting = 1;
    
elseif strcmp('dam',domain)

    mdp_vars = dam_mdpvariables();
    k0 = [50; -50; 0; 0; 50];
    dim = mdp_vars.nvar_action;
%     policy = gaussian_policy(@dam_basis_rbf, dim, k0, 50);
%     policy = smart_gaussian_policy(@dam_basis_rbf, dim, k0, 20);
    policy = logistic_gaussian_policy(@dam_basis_rbf, dim, k0, 1, 50);
    episodes = 1;
    steps = 100;
    gamma = 1;
    avg_rew_setting = 1;

elseif strcmp('lqr',domain)

    mdp_vars = lqr_mdpvariables();
    dim = mdp_vars.nvar_action;
    k0 = -0.5 * eye(dim);
    s0 = 1 * eye(dim);
    offset0 = zeros(dim,1);
    tau = 1.3 * ones(size(diag(s0)));
%     policy = logistic_gaussian_policy(@lqr_basis_pol, dim, k0, diag(s0), tau);
%     policy = smart_gaussian_policy(@lqr_basis_pol, dim, k0, s0);
%     policy = smart_diag_gaussian_policy(@lqr_basis_pol, dim, k0, diag(s0));
%     policy = full_smart_gaussian_policy(@lqr_basis_pol, dim, offset0, k0, (s0));
    policy = gaussian_policy(@lqr_basis_pol, dim, k0, s0);
    episodes = 1;
    steps = 50;
    gamma = 0.9;
    avg_rew_setting = 0;

else
    
    error('Domain unknown')

end

n_obj = mdp_vars.nvar_reward;
max_obj = mdp_vars.max_obj;

end
