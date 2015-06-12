function mdp_vars = lqr_mdpvariables()

dim = 2;
mdp_vars.dim = dim;
mdp_vars.nvar_state = dim;
mdp_vars.nvar_action = dim;
mdp_vars.nvar_reward = dim;
mdp_vars.max_obj = ones(dim,1);
mdp_vars.gamma = 0.9;
mdp_vars.is_avg = 0;
mdp_vars.isStochastic = 0;

return