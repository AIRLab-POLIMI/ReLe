function mdp_vars = Como_mdpvariables()

dim = 2; % step cost, objectives
mdp_vars.nvar_state = 2; % number of state variables
mdp_vars.nvar_action = 1; % only one controler/control variable
mdp_vars.nvar_reward = dim; % see above
mdp_vars.max_obj = [1; 1; 1; 1]; % maximum, worst (hypothetical) values of the objectives -> opposite of the utopia point? 
mdp_vars.max_obj = mdp_vars.max_obj(1:dim); % use only two of the above max values of objectives
mdp_vars.gamma = 1; % Discount factor, 1= no discount rate 
mdp_vars.isAvg = 1; % ? 
mdp_vars.isStochastic = false; % 1; % random init state and random inflow

% 1 to penalize the policy when it violates the problem's constraints
mdp_vars.penalize = 0; % no additional penalty for violation of the boundary conditions. 

return
