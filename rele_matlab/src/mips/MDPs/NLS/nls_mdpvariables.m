function mdp_vars = nls_mdpvariables()

mdp_vars.nvar_state   = 2;
mdp_vars.nvar_action  = 1;
mdp_vars.nvar_reward  = 1;
mdp_vars.max_obj      = 1;
mdp_vars.gamma        = 0.95;
mdp_vars.is_avg       = 0;
mdp_vars.isStochastic = 0;


mdp_vars.noise_mean = 0.0;
mdp_vars.noise_std  = 0.02;
mdp_vars.pos0_mean  = 1.0;
mdp_vars.pos0_std   = 0.001;
mdp_vars.reward_reg = 0.1;

return
