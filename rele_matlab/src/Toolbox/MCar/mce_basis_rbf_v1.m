function phi = mce_basis_rbf_v1(state,action)


n_centers = 4;
if nargin == 0
    phi = 3 * (basis_rbf(n_centers,[-2 1; -4 4]) + 1);
else
    Phi = [1; basis_rbf(n_centers,[-2 1; -4 4],state)];
    n = length(Phi);
    phi = zeros(3*n,1);
    i = action - 1;
    phi(n*i+1:n*i+n) = Phi;
end 

end