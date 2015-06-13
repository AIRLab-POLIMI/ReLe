function mdp_vars = uwv_mdpvariables()

mdp_vars.nvar_state = 1;
mdp_vars.nvar_action = 1;
mdp_vars.nvar_reward = 1;
mdp_vars.max_obj = 1;
mdp_vars.gamma = 0.99;
mdp_vars.is_avg = 0;
mdp_vars.isStochastic = 0;


mdp_vars.dt = 0.03;
mdp_vars.vel_lo = -5;
mdp_vars.vel_hi = 5;
mdp_vars.trust_lo = -30;
mdp_vars.trust_hi = 30;
mdp_vars.mu = 0.3;
mdp_vars.setPoint = 4;
mdp_vars.C = 0.01;

mdp_vars.action_list = [1,2,3,4,5];
mdp_vars.action_values = [mdp_vars.trust_lo, mdp_vars.trust_lo/2, ...
    0, mdp_vars.trust_hi/2, mdp_vars.trust_hi];

return
