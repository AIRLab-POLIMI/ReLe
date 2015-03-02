function  [nextstate, reward, absorb] = resource_simulator(state, action)

mdp_vars = resource_mdpvariables();

if nargin == 0
    
    nextstate = [5; 3; 0; 0];
    return
    
elseif nargin == 1
    
    nextstate = state;
    return
    
end

i = state(1);
j = state(2);

switch action
    case 1
        j1 = max(1,j-1);
        nextstate = [i; j1];
    case 2
        j2 = min(mdp_vars.state_dim(2),j+1);
        nextstate = [i; j2];
    case 3
        i3 = max(1,i-1);
        nextstate = [i3; j];
    case 4
        i4 = min(mdp_vars.state_dim(1),i+1);
        nextstate = [i4; j];
    otherwise
        nextstate = state;
end

nextstate(3) = state(3);
nextstate(4) = state(4);
if nextstate(1) == 1 && nextstate(2) == 3
    nextstate(3) = 1;
elseif nextstate(1) == 2 && nextstate(2) == 5
    nextstate(4) = 1;
end

reward1 = 0;
if resource_check_fight(nextstate)
    reward1 = -1; % fight
    state(3) = 0;
    state(4) = 0;
    nextstate(1) = 5;
    nextstate(2) = 3;
end

absorb = 0;

reward2 = 0;
reward3 = 0;
if(nextstate(1) == 5 && nextstate(2) == 3)
    reward2 = state(3); % gold
    reward3 = state(4); % gems
    nextstate(3) = 0;
    nextstate(4) = 0;
end

reward = [reward1; reward2; reward3] ./ mdp_vars.max_obj;

return
