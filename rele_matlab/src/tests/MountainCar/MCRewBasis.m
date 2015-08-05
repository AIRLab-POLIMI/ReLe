function V = MCRewBasis(X, center, sigma)
A = -(X(:,1)-center(:,1)).^2/sigma(1);
B = -(X(:,2)-center(:,2)).^2/sigma(2);
V = exp(A + B);
end