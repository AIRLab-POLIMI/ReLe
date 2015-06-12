function [nextState, reward, absorb] = heat_simulator(state, action)
% Mulit-Heat system with N-rooms


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% System parameter
%--------------------------------------

mdp_vars = heat_mdpvariables()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check meaning of the function
%--------------------------------------
if nargin == 0
    % draw initial state
    % MODIFICARE
    nextState = [-0.5 0]';
    return
    
elseif nargin == 1
    % DO NOT TOUCH
    nextState = state;
    return
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update state
%--------------------------------------

nextState = ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute reward
%--------------------------------------

reward = 
absorb =

end % heat_simulator()

