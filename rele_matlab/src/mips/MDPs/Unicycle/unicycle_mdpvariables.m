function mdp_vars = unicycle_mdpvariables()

mdp_vars.nvar_state = 3;
mdp_vars.nvar_action = 2;
mdp_vars.nvar_reward = 1;
mdp_vars.max_obj = 1;
mdp_vars.gamma = 0.99;
mdp_vars.is_avg = 0;
mdp_vars.isStochastic = 0;


mdp_vars.dt = 0.03;
mdp_vars.reward_th = 0.1;

return
