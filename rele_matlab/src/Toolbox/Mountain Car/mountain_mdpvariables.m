function mdp_vars = mountain_mdpvariables()

mdp_vars.nvar_state = 2;
mdp_vars.nvar_action = 1;
mdp_vars.nvar_reward = 3;
mdp_vars.action_list = [1,2,3];
mdp_vars.max_obj = [1; 1; 1];
mdp_vars.gamma = 1;
mdp_vars.is_avg = 0;
mdp_vars.isStochastic = 0;

return