function phi = nls_cov_v1(state)

if nargin == 0
    phi = 2;
else
    phi = 0.5 * ones(size(state))' * state;
end

end