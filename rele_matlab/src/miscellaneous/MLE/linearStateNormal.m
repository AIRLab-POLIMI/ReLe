function y = linearStateNormal(data, w, s, DS)
sigma = max(1e-8,s);
mu = w * DS;
y = max(1e-8, normpdf(data,mu,sigma));
end