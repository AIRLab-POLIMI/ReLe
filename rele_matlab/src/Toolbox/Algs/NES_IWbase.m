function [nat_grad, stepsize] = NES_IWbase (pol_high, J, Theta, W, lrate)

n_episodes = length(J);

num = 0;
den = 0;
dlogPidtheta = zeros(pol_high.dlogPidtheta,n_episodes);

% Compute optimal baseline
for k = 1 : n_episodes
    
    dlogPidtheta(:,k) = pol_high.dlogPidtheta(Theta(:,k));
    
    num = num + dlogPidtheta(:,k).^2 * J(k) * W(k)^2;
    den = den + dlogPidtheta(:,k).^2 * W(k)^2;
    
end

b = num ./ den;
b(isnan(b)) = 0;
% b = mean(J);

% Estimate gradient and Fisher information matrix
grad = 0;
F = 0;
for k = 1 : n_episodes
    grad = grad + dlogPidtheta(:,k) .* (J(k) - b) * W(k);
    F = F + dlogPidtheta(:,k) * dlogPidtheta(:,k)' * W(k);
end
grad = grad / sum(W);
F = F / sum(W);

% If we can compute the FIM in closed form, use it
if ismethod(pol_high,'fisher')
    F = pol_high.fisher;
end

% If we can compute the FIM inverse in closed form, use it
if ismethod(pol_high,'inverseFisher')
    invF = pol_high.inverseFisher;
    nat_grad = invF * grad;
elseif rank(F) == size(F,1)
    nat_grad = F \ grad;
else
    warning('Fisher matrix is lower rank (%d instead of %d).', rank(F), size(F,1));
    nat_grad = pinv(F) * grad;
end

if nargin >= 5
    lambda = sqrt(grad' * nat_grad / (4 * lrate));
    lambda = max(lambda,1e-8); % to avoid numerical problems
    stepsize = 1 / (2 * lambda);
end

end