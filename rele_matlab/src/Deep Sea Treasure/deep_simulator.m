function  [nextstate, reward, absorb] = deep_simulator(state, action)

mdp_vars = deep_mdpvariables();

if nargin == 0
    
    nextstate = [1; 1];
    return
    
elseif nargin == 1
    
    nextstate = state;
    return
    
end

i = state(1);
j = state(2);

switch action
    % left
    case 1
        j1 = max(1,j-1);
        if ~deep_check_black(i,j1)
            j1 = j;
        end
        nextstate = [i; j1];
    % right
    case 2
        j2 = min(mdp_vars.state_dim(2),j+1);
        if ~deep_check_black(i,j2)
            j2 = j;
        end
        nextstate = [i; j2];
    % up
    case 3
        i3 = max(1,i-1);
        if ~deep_check_black(i3,j)
            i3 = i;
        end
        nextstate = [i3; j];
    % down
    case 4
        i4 = min(mdp_vars.state_dim(1),i+1);
        if ~deep_check_black(i4,j)
            i4 = i;
        end
        nextstate = [i4; j];
    otherwise
        nextstate = state;
end

% treasure value
reward1 = deep_reward_treasure(nextstate);
% time
reward2 = -1;
if reward1 == 0
    absorb = 0;
else
    absorb = 1;
end
reward = [reward1; reward2] ./ mdp_vars.max_obj;

return
