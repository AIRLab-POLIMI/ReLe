function mdp_vars = lqr_mdpvariables()

dim = 2;
mdp_vars.dim = dim;
mdp_vars.nvar_state = dim;
mdp_vars.nvar_action = dim;
mdp_vars.nvar_reward = dim;
mdp_vars.max_obj = ones(dim,1);

return