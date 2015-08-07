function [nextState, reward, absorb] = nls_simulator(state, action)
% Segway simulater


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% System parameter
%--------------------------------------
mdp_vars = nls_mdpvariables();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check meaning of the function
%--------------------------------------
if nargin == 0
    % draw initial state
    nextState = normrnd(mdp_vars.pos0_mean, mdp_vars.pos0_std);
    return
    
elseif nargin == 1
    % DO NOT TOUCH
    nextState = state;
    return
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update state
%--------------------------------------
model_noise = normrnd(mdp_vars.noise_mean, mdp_vars.noise_std);

nexstate = ones(2,1);
nextState(2) = state(2) + 1.0/(1 + exp(-action)) - 0.5 + model_noise;
nextState(1) = state(1) - 0.1 * nextState(2) + model_noise;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute reward
%--------------------------------------
norm2state = norm(state,2);
if norm2state < mdp_vars.reward_reg
    reward = 1.0;
else
    reward = 0.0;
end
absorb = 0;

end % nls_simulator()

