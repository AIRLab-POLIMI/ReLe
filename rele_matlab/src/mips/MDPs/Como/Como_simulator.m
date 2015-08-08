function [nextstate, reward, absorb] = Como_simulator(state, action)

persistent Como_Inflow sys_param

if isempty(Como_Inflow)
    %%% Load inflow observations
    load larioInflow
    Como_Inflow = larioInflow;
end

if isempty(sys_param)
    load Lario_param_denser
    % % discretization of the storage:
    sys_param.discr_s = Lario_param.como.discr_s' ; % (n_s,1)

    % % minimum and maximum release functions:
    sys_param.min_rel = Lario_param.como.min_rel' ; % (n_s,n_e)
    sys_param.max_rel = Lario_param.como.max_rel' ; % (n_s,n_e)

    % % discretization of the inflow:
    sys_param.discr_inflow = Lario_param.como.discr_a' ; % (n_e,1)
end

mdp_vars = Como_mdpvariables();

env = Como_environment();
reward = zeros(mdp_vars.nvar_reward,1);

% Initial states
day_init = Como_Inflow.time(1,1);
tnat_init = Como_Inflow.tnat(1,1);
s_init = Como_Inflow.value(Como_Inflow.tnat == tnat_init, 1); % levels of 1st of Jan from 1945 to 2013

if nargin == 0
    
    if ~mdp_vars.penalize %% if penalize is set to 0 (false) enter here
        idx = randi(length(s_init), 1);
        nextstate(1) = s_init(idx);
    else
        nextstate(1) = unifrnd(0,160); % this doesn't make any sense now
    end
    nextstate(2) = day_init;
    return
    
elseif nargin == 1 % you only put in state: If you don't act, the state does not change (which is strange) ?
    
    nextstate = state;
    return
    
end

%% From Simona's code sim_dp_lario

storage = state(1) * 145.9 * 10^6; % state(1) is the como lake level
idx_today = state(2) - Como_Inflow.time(1,1) + 1; % state(2) is the datenum
inflow = Como_Inflow.value(idx_today, 3);

% Minimum and maximum release for current storage and inflow:
[ ~ , idx_inflow ] = min( abs( sys_param.discr_inflow - inflow ) ) ;
min_action = interp_lin_scalar( sys_param.discr_s , sys_param.min_rel( : , idx_inflow ) , storage ) ;% sys_param.v = interp_lin_scalar( sys_param.discr_s , sys_param.min_rel( : , idx_e ) , s(t) ) ;
max_action = interp_lin_scalar( sys_param.discr_s , sys_param.max_rel( : , idx_inflow ) , storage ) ;  %sys_param.V = interp_lin_scalar( sys_param.discr_s , sys_param.max_rel( : , idx_e ) , s(t) ) ;

%%%% Not required in this context
% Find the set of optimal decisions:
%sys_param.t_nat   = t_nat(t) ;
%sys_param.t_ant   = t_ant(t) ;
% [ H_min , idx_u ] = Bellman_det_Lario( H(:,t+1) , s(t) , e_sim(t+1) , sys_param ) ;
% Choose one decision value (Extractor):
%u(t) = extractor_ref( idx_u , sys_param.discr_u , sys_param.demand(t_nat(t)) ) ;
% Compute the release:
% r(t+1)   = min( sys_param.V , max( sys_param.v , u(t) ) ) ;
% Compute future storage:
% s(t+1) = s(t) + delta * ( e_sim(t+1) - r(t+1) ) ;


% Bound the action - replaced by lines 37/38
% min_action = max(state - env.S_MIN_REL, 0);
% max_action = state;

penalty = 0;

if min_action > action || max_action < action
    
    % Give the penalty only during the learning
    penalty = -max(action - max_action, min_action - action) * mdp_vars.penalize;
    action  = max(min_action, min(max_action, action));
    
end

% Transition dynamic
nextstate(1) = state(1) + env.delta * (inflow - action);
nextstate(2) = state(2) + 1;

% Cost due to the excess level w.r.t. a flooding threshold (upstream)
% reward(1) = -max(nextstate(1)/env.S - env.H_FLO_U, 0) + penalty;
reward(1) = nextstate(1)/env.S > env.H_FLO_U;
reward(1) = reward(1) * 365; % to get average flooded days per year

% Deficit in the water supply w.r.t. the water demand
t_nat = Como_Inflow.tnat(idx_today);
reward(2) = -max(env.W_IRR(t_nat) - action, 0) + penalty;


% q = 0;
% if action > env.Q_MEF
%     q = action - env.Q_MEF;
% end
%
% p_hyd = env.ETA * env.G * env.GAMMA_H2O * nextstate(1)/env.S * q / (3.6e6);
%
% % Deficit in the hydroelectric supply w.r.t the hydroelectric demand
% if mdp_vars.nvar_reward == 3
%     reward(3) = -max(env.W_HYD - p_hyd, 0) + penalty;
% end
%
% % Cost due to the excess level w.r.t. a flooding threshold (downstream)
% if mdp_vars.nvar_reward == 4
%     reward(4) = -max(action - env.Q_FLO_D, 0) + penalty;
% end

absorb = 0;

reward = reward ./ mdp_vars.max_obj;

return
