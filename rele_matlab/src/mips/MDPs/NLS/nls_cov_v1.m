function phi = nls_cov_v1(state)

if nargin == 0
    phi = 1;
else
    phi = 0.5 * sum(state);
    phi = phi^2;
end

end