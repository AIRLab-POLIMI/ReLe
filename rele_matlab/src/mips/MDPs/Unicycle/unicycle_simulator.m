function [nextState, reward, absorb] = unicycle_simulator(state, action)
% Mulit-Heat system with N-rooms


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% System parameter
%--------------------------------------

mdp_vars = unicycle_mdpvariables();
dt = mdp_vars.dt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check meaning of the function
%--------------------------------------
if nargin == 0
    % draw initial state
    goal = [0;0;0];
    
    x  = -4 + 8 * rand(1);
    y  = -4 + 8 * rand(1);
    th = -pi + 2 * pi * rand(1);
    
    Tr = [cos(0), sin(0), 0;
        -sin(0), cos(0), 0;
        0, 0, 1];
    e = Tr *[x;y;th];
    
    s0 = ones(3,1);
    s0(1) = sqrt(e(1)^2+e(2)^2);
    s0(2) = atan2(e(2),e(1)) - e(3) + pi;
    s0(3) = s0(2) + e(3);
    
    s0(2) = wrapToPi(s0(2));
    s0(3) = wrapToPi(s0(3));
    
    nextState = s0;
    return
    
elseif nargin == 1
    % DO NOT TOUCH
    nextState = state;
    return
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update state
%--------------------------------------

[t,y] = ode45(@(t,s) unicycle_ode(t, s, action), [0 dt], state);
nextState = y(end,:)';

nextState(2) = wrapToPi(nextState(2));
nextState(3) = wrapToPi(nextState(3));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute reward
%--------------------------------------
v = action(1);
w = action(2);
dist = (abs(nextState(1)) + 10 * abs(nextState(2)) + 10 * abs(nextState(3)));
reward = - dist - 0.1 * w * w - 0.05 * v * v;

absorb = 0;
if (dist < mdp_vars.reward_th)
    absorb = 1;
end

end % unicycle_simulator()

