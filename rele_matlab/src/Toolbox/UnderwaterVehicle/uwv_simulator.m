function [nextState, reward, absorb] = uwv_simulator(state, action)
% Mulit-Heat system with N-rooms


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% System parameter
%--------------------------------------

mdp_vars = uwv_mdpvariables();
vel_lo = mdp_vars.vel_lo;
vel_hi = mdp_vars.vel_hi;
dt = mdp_vars.dt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check meaning of the function
%--------------------------------------
if nargin == 0
    % draw initial state
    nextState = vel_lo + rand() * (vel_hi - vel_lo);
    return
    
elseif nargin == 1
    % DO NOT TOUCH
    nextState = state;
    return
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update state
%--------------------------------------
ac = mdp_vars.action_values(action);

[t,y] = ode45(@(t,s) uwv_ode(t,s,ac), [0 dt], state);
nextState = y(end); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute reward
%--------------------------------------
if abs(mdp_vars.setPoint - nextState) < mdp_vars.mu
    reward = 0.0;
else
    reward = -mdp_vars.C;
end
absorb = 0;

end % uwv_simulator()

