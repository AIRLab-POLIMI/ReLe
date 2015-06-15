function mdp_vars = heat_mdpvariables()

dim = 2;
mdp_vars.nvar_state = dim+1;
mdp_vars.nvar_action = 1;
mdp_vars.nvar_reward = 1;
mdp_vars.action_list = 1:(dim+1);
mdp_vars.max_obj = 1;
mdp_vars.gamma = 0.9;
mdp_vars.is_avg = 0;
mdp_vars.isStochastic = 0;

mdp_vars.Ta = 6;
mdp_vars.dt = 0.1;
mdp_vars.a = 0.8;
mdp_vars.s2n = 1;
mdp_vars.TUB = 22;
mdp_vars.TLB = 17.5;

mdp_vars.Nr = dim;

mdp_vars.A = 0.33*mdp_vars.dt * (diag(ones(dim-1,1),1) + diag(ones(dim-1,1),-1));
mdp_vars.B = 0.25*mdp_vars.dt * ones(dim,1);
mdp_vars.C = 12*mdp_vars.dt * ones(dim,1);
mdp_vars.Xi = mdp_vars.A - diag(mdp_vars.B+sum(mdp_vars.A,2));
mdp_vars.Gam = mdp_vars.B * mdp_vars.Ta;


return
