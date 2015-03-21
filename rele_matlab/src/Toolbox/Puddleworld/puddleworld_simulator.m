function  [nextstate, reward, absorb] = ...
    puddleworld_simulator(state, action)

env = puddleworld_environment();
mdp_vars = puddleworld_mdpvariables();

if nargin == 0

    while true
        nextstate = [rand; rand];
        if ~(nextstate(1) > 0.95 && nextstate(2) > 0.95)
            break;
        end
    end
    return
    
elseif nargin == 1
    
    nextstate = state;
    return
    
end

switch action
    % left
    case 1
        if state(1) - env.step < env.xmin
            state(1) = env.xmin;
        else
            state(1) = state(1) - env.step;
        end
    % right
    case 2
        if state(1) + env.step > env.xmax
            state(1) = env.xmax;
        else
            state(1) = state(1) + env.step;
        end
    % up
    case 3
        if state(2) + env.step > env.ymax
            state(2) = env.ymax;
        else
            state(2) = state(2) + env.step;
        end
    % down
    case 4
        if state(2) - env.step < env.ymin
            state(2) = env.ymin;
        else
            state(2) = state(2) - env.step;
        end
    otherwise
end

nextstate = state;

% distance from the nearest edge of the puddle
reward1 = puddleworld_reward_distance(nextstate);
% time
reward2 = -1;

if nextstate(1) > 0.95 && nextstate(2) > 0.95
    absorb = 1;
else
    absorb = 0;
end

reward = [reward1; reward2] ./ mdp_vars.max_obj;

return