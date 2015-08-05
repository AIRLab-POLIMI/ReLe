function phi = uwv_basis_pol_v1(state,action)
deg = 2;
mdp_vars = uwv_mdpvariables();
numfeatures = basis_poly(deg,1,0);
% The basis functions are repeated for each action
numbasis = numfeatures * (length(mdp_vars.action_list) - 1);
if nargin < 1
    phi = numbasis;
else
    phi = zeros(numbasis,1);
    init_idx = numfeatures;
    i = action - 1;
    phi(init_idx*i+1:init_idx*i+init_idx) = basis_poly(deg,1,0,state);
end
end