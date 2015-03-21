function mdp_vars = dam_mdpvariables()

dim = 2;
mdp_vars.nvar_state = 1;
mdp_vars.nvar_action = 1;
mdp_vars.nvar_reward = dim;
if dim == 3
    mdp_vars.max_obj = [20; 20; 2];
elseif dim == 2
    mdp_vars.max_obj = [20; 20];
else
    mdp_vars.max_obj = ones(dim,1);
end

mdp_vars.evaluation = 0; % 1 if we want to evaluate policies, 0 if we want to learn

return