function phi = heat_basis_rbf_v1(s,a)
	
    mdp_vars = heat_mdpvariables();
    nb = 5;
    range = [13.5 26];
    
    ndim = mdp_vars.Nr;
    nactions = length(mdp_vars.action_list);
	nbasis = nb^ndim;

	nmodes = ndim+1;
% 	nmodes = 1;		% Da commentare in caso di distinzione tra i modi
	
	phi = zeros(nbasis*nmodes*(nactions-1),1);
    if nargin < 1
        phi = length(phi);
		return
    end
	

	if nmodes ~= ndim+1
		s(1) = 0;
	end
	phi((1:nbasis)+s(1)*nbasis+a*(nbasis*nmodes),1) = basis_rbf(nb,repmat(range,ndim,1),s(2:end));

end