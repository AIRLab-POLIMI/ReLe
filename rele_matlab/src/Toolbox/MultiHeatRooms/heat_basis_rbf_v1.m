function phi = heat_basis_rbf_v1(s,a)

% 	ndim = 2;
% 	nb = 5;
% 	range = [14.5 25];
	
    mdp_vars = heat_mdpvariables()
    
	nmodes = ndim+1;
	nactions = ndim+1;
	nbasis = nb^ndim;

	nmodes = 1;		% Da decommentare in caso di non distinzione tra i modi
	
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