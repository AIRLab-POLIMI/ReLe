function [nextState, reward, absorb] = mce_simulator(state, action)
% Simulatore Mountain Car secondo Ernst (Tree-based, 2005)

if nargin == 0
    % initial state
    nextState = [-0.5 0]';
    return
    
elseif nargin == 1
    
    nextState = state;
    return
    
end

%%% ----------------------------------------------------------------------------------------------
%%% Mountain Car model initialization
%%% ----------------------------------------------------------------------------------------------

model.dt = 0.1;   % Passo d'integrazione
model.mass = 1;   % massa
model.g = 9.81;   % accellerazione di gravita'

% Limiti intervallo della posizione
model.xLB = -2;
model.xUB = 1;

% Limiti intervallo della velocita
model.vLB = -4;
model.vUB = 4;

% Azioni disponibili
model.throttle = [-4 0 4]';

%%% ----------------------------------------------------------------------------------------------
%%% Parse input argument
%%% ----------------------------------------------------------------------------------------------
position = state(1);
velocity = state(2);
throttle = model.throttle(action);

%%% ----------------------------------------------------------------------------------------------
%%% Update state
%%% ----------------------------------------------------------------------------------------------
psecond = ddp(model, position, velocity, throttle);

pNext = position + model.dt * velocity + 0.5 * model.dt * model.dt * psecond;
vNext = velocity + model.dt * psecond;
nextState = [pNext vNext]';

%%% ----------------------------------------------------------------------------------------------
%%% Compute reward signal
%%% ----------------------------------------------------------------------------------------------
[reward, absorb] = mountainCar_reward_ernst(model, pNext, vNext);

    %%% -------------------------------------------------------------------------------------------
    %%% Helper functions
    %%% -------------------------------------------------------------------------------------------

    %%% Funzione rappresentante la hill
    function hill_val = hill(pos)

        hill_val = (pos <=0) .* (pos.^2 + pos) + (pos >0) .* ( pos./sqrt(1+5*pos.^2));

    return % hill()

    %%% Derivata della funzione rappresentante la hill
    function dhill_val = dhill(pos)

        if (pos < 0.0)
            dhill_val = 2*pos + 1;
        else
            dhill_val = 1.0/sqrt(1+5*pos*pos) - 5*pos*pos / (1+5*pos*pos)^1.5;
        end

    return % dhill()

    %%% Derivata seconda della posizione
    function ddp_val = ddp(model, pos, velocity, throttle)

        A = throttle / ( model.mass * (1 + dhill(pos)*dhill(pos)) );
        B = model.g * dhill(pos) / ( 1 + dhill(pos)*dhill(pos) );
        C = velocity^2 * dhill(pos) * dhill(dhill(pos)) / (1 + dhill(pos)^2);
        ddp_val = A - B - C;

    return % ddp()

    % Funzione di reward
    function [reward, absorb] = mountainCar_reward_ernst(model, position, velocity)

        reward = 0;
        absorb = false;
        if (position < model.xLB) || (abs(velocity) > model.vUB)
            reward = -1;
            absorb = true;
        elseif (position > model.xUB) && (abs(velocity) <= model.vUB)
            reward = 1;
            absorb = true;
        end

    return % mountainCar_reward_ernst()

return % mc_simulator()

