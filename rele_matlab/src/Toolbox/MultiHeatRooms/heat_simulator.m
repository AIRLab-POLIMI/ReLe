function [nextState, reward, absorb] = heat_simulator(state, action)
% Mulit-Heat system with N-rooms


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% System parameter
%--------------------------------------

mdp_vars = heat_mdpvariables();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check meaning of the function
%--------------------------------------
if nargin == 0
    % draw initial state
    nextState = [0;(26-13.5)*rand(mdp_vars.Nr,1)+13.5];
    return
    
elseif nargin == 1
    % DO NOT TOUCH
    nextState = state;
    return
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update state
%--------------------------------------

nextState=state;

if action~=state(1) && rand<=mdp_vars.a
    nextState(1) = action;
end

    
noise = randn(mdp_vars.Nr,1) * sqrt(mdp_vars.s2n * mdp_vars.dt);

nextState(2:end) = (eye(mdp_vars.Nr)+mdp_vars.Xi) * state(2:end) + mdp_vars.Gam + noise;

if nextState(1)>1   
    nextState(nextState(1)) = nextState(nextState(1)) + mdp_vars.C(nextState(1)-1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute reward
%--------------------------------------

reward = sum(max((nextState(2:end)-mdp_vars.TUB).*(nextState(2:end)-mdp_vars.TLB),0));
absorb = false;

end % heat_simulator()

