function phi = dam_basis_pol(state)

if nargin == 0
    phi = 4;
    return
end

phi = [1; state; state^2; state^3];

return
