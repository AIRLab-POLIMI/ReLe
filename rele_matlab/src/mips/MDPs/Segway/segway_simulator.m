function [nextState, reward, absorb] = segway_simulator(state, action)
% Segway simulater


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% System parameter
%--------------------------------------

mdp_vars = segway_mdpvariables();
dt = mdp_vars.dt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check meaning of the function
%--------------------------------------
if nargin == 0
    % draw initial state
    nextState = [0.08; 0; 0];
    return
    
elseif nargin == 1
    % DO NOT TOUCH
    nextState = state;
    return
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update state
%--------------------------------------
ac = action;

[t,y] = ode45(@(t,s) segway_ode(t,s,ac), [0 dt], state);
nextState = y(end,:)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute reward
%--------------------------------------
if abs(nextState[1]) > pi/18
    reward = -1000;
    absorb = 1;
else
    lqr_cost = nextState'*nextState;% + action'*action;
    reward = -lqr_cost;
    absorb = 0;
end

end % uwv_simulator()

