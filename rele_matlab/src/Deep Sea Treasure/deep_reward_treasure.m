function value = deep_reward_treasure(state)

mdp_vars = deep_mdpvariables;
reward = zeros(mdp_vars.state_dim(1),mdp_vars.state_dim(2));
reward(2,1) = 1;
reward(3,2) = 2;
reward(4,3) = 3;
reward(5,4) = 5;
reward(5,5) = 8;
reward(5,6) = 16;
reward(8,7) = 24;
reward(8,8) = 50;
reward(10,9) = 74;
reward(11,10) = 124;
value = reward(state(1),state(2));

return
