function [n_obj, n_params, mu0, sigma0] = settings_episodic( domain, isDet )
% Inputs: 
% - domain   : the name of the MDP
% - isDet    : 1 if the low-level policy is deterministic, 0 otherwise
%
% Outputs:
%
% - n_obj    : number of objectives of the MDP
% - n_params : number of parameters of the low-level policy to be learnt
% - mu0      : initial mean of the high-level policy (distribution to draw 
%              low-level policy parameters)
% - sigma0   : initial covariance of the high-level policy

[n_obj, pol_low] = settings(domain);

n_params = size(pol_low.theta,1) - pol_low.dim_variance_params * isDet;

mu0 = pol_low.theta(1:n_params);

if strcmp('deep',domain)

    sigma0 = 10 * eye(n_params);
    
elseif strcmp('dam',domain)

    sigma0 = 100 * eye(n_params);

elseif strcmp('lqr',domain)

    sigma0 = 0.1 * eye(n_params);

else
    
    error('Domain unknown.')

end

sigma0 = sigma0 + diag(abs(pol_low.theta(1:n_params)));

end
