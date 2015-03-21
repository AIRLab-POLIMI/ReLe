function  [nextstate, reward, absorb] = dam_simulator(state, action)

mdp_vars = dam_mdpvariables();

env = dam_environment();
reward = zeros(mdp_vars.nvar_reward,1);

% initial states according to:
% Castelletti et al, Tree-based fitted q-iteration for MOMDP
s_init = [9.6855361e+01, 5.8046026e+01, ...
    1.1615767e+02, 2.0164311e+01, ...
    7.9191000e+01, 1.4013098e+02, ...
    1.3101816e+02, 4.4351321e+01, ...
    1.3185943e+01, 7.3508622e+01, ...
    ];

if nargin == 0
    
    if mdp_vars.evaluation
        idx = randi(length(s_init), 1); % init state for evaluation
        nextstate = s_init(idx);
    else
        nextstate = unifrnd(0,160); % init state for learning
    end
    return
    
elseif nargin == 1
    
    nextstate = state;
    return
    
end

% bound the action
min_action = max(state - env.S_MIN_REL, 0);
max_action = state;
penalty = 0;

if min_action > action || max_action < action

    % take the penalty only during the learning
    penalty = -max(action - max_action, min_action - action) * ~mdp_vars.evaluation;
    action  = max(min_action, min(max_action, action));
    
end

% transition dynamic
nextstate  = state + env.DAM_INFLOW - action;

% cost due to the excess level w.r.t. a flooding threshold (upstream)
reward(1) = -max(nextstate/env.S - env.H_FLO_U, 0) + penalty;

% deficit in the water supply w.r.t. the water demand
reward(2) = -max(env.W_IRR - action, 0) + penalty;

q = 0;
if action > env.Q_MEF
    q = action - env.Q_MEF;
end
p_hyd = env.ETA * env.G * env.GAMMA_H2O * nextstate/env.S * q / (3.6e6);

% deficit in the hydroelectric supply w.r.t to hydroelectric demand
if mdp_vars.nvar_reward == 3
    reward(3) = -max(env.W_HYD - p_hyd, 0) + penalty;
end

% cost due to the excess level w.r.t. a flooding threshold (downstream)
if mdp_vars.nvar_reward == 4
    reward(4) = -max(action - env.Q_FLO_D, 0) + penalty;
end
absorb = 0;

reward = reward ./ mdp_vars.max_obj;

return
