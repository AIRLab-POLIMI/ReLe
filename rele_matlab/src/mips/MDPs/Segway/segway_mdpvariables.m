function mdp_vars = segway_mdpvariables()

mdp_vars.nvar_state   = 3;
mdp_vars.nvar_action  = 1;
mdp_vars.nvar_reward  = 1;
mdp_vars.max_obj      = 1;
mdp_vars.gamma        = 0.99;
mdp_vars.is_avg       = 0;
mdp_vars.isStochastic = 0;


mdp_vars.dt  = 0.03;
mdp_vars.Mp  = 10;
mdp_vars.Mr  = 15;
mdp_vars.Ip  = 19;
mdp_vars.Ir  = 19;
mdp_vars.l   = 1.2; %m
mdp_vars.r   = 0.2;
mdp_vars.g   = 9.81;

return
