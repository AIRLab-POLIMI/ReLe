function mdp_vars = puddleworld_mdpvariables()

mdp_vars.nvar_state = 2;
mdp_vars.nvar_action = 1;
mdp_vars.nvar_reward = 2;
mdp_vars.action_list = [1,2,3,4];
mdp_vars.max_obj = [1; 1];

return