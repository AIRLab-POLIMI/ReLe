function mdp_vars = dam_mdpvariables()

dim = 2;
mdp_vars.nvar_state = 1;
mdp_vars.nvar_action = 1;
mdp_vars.nvar_reward = dim;
mdp_vars.max_obj = [20; 20; 2; 1];
mdp_vars.max_obj = mdp_vars.max_obj(1:dim);
mdp_vars.gamma = 1;
mdp_vars.is_avg = 1;
mdp_vars.isStochastic = 1; % random init state and random inflow

% 1 to penalize the policy when it violates the problem's constraints
mdp_vars.penalize = 0;

return
