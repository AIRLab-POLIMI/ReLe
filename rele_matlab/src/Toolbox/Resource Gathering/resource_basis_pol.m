function phi = resource_basis_pol(state, action)

mdp_vars = deep_mdpvariables();

numfeatures = 24;

%%% The basis functions are repeated for each action
numbasis = numfeatures * (length(mdp_vars.action_list) - 1);

%%% If no arguments just return the number of basis functions
if nargin == 0
    phi = numbasis;
    return
end

tmp = zeros(numfeatures,1);
tmp(1) = 1;

tmp(2) = state(1) * state(3) * state(4);
tmp(3) = state(1) * ~state(3) * state(4);
tmp(4) = state(1) * state(3) * ~state(4);
tmp(5) = state(1) * ~state(3) * ~state(4);

tmp(6) = state(2) * state(3) * state(4);
tmp(7) = state(2) * ~state(3) * state(4);
tmp(8) = state(2) * state(3) * ~state(4);
tmp(9) = state(2) * ~state(3) * ~state(4);

tmp(10) = state(1) * state(2) * state(3) * state(4);
tmp(11) = state(1) * state(2) * ~state(3) * state(4);
tmp(12) = state(1) * state(2) * state(3) * ~state(4);
tmp(13) = state(1) * state(2) * ~state(3) * ~state(4);

tmp(14) = (state(1) == 5 && state(2) == 3);
tmp(15) = (state(1) == 1 && state(2) == 3);
tmp(16) = (state(1) == 2 && state(2) == 5);

tmp(17) = (state(1) == 5) * state(3) * state(4);
tmp(18) = (state(1) == 5) * ~state(3) * state(4);
tmp(19) = (state(1) == 5) * state(3) * ~state(4);
tmp(20) = (state(1) == 5) * ~state(3) * ~state(4);
tmp(21) = (state(2) == 5) * state(3) * state(4);
tmp(22) = (state(2) == 5) * ~state(3) * state(4);
tmp(23) = (state(2) == 5) * state(3) * ~state(4);
tmp(24) = (state(2) == 5) * ~state(3) * ~state(4);

%%% Features depending only from the state
if nargin == 1
    phi = tmp;
    return
end

%%% Initialize
phi = zeros(numbasis,1);

%%% Find the starting position
base = numfeatures;

i = action - 1;
phi(base*i+1:base*i+base) = tmp;

return
