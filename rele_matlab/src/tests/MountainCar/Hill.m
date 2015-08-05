function h = Hill(X)
h = zeros(size(X));
h(X < 0) = X(X<0).^2 + X(X<0);
h(X>=0) = X(X>=0)./sqrt(1+5*X(X>=0).^2);
end