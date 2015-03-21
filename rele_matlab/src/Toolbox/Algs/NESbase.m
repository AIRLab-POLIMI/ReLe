%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: D Wierstra, T Schaul, T Glasmachers, Y Sun, J Peters, 
% J Schmidhuber (2014)
% Natural Evolution Strategy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nat_grad = NESbase (pol_high, J, Theta, lrate)

n_episodes = length(J);

num = 0;
den = 0;
dlogPidtheta = zeros(pol_high.dlogPidtheta,n_episodes);

% Compute optimal baseline
for k = 1 : n_episodes
    
    dlogPidtheta(:,k) = pol_high.dlogPidtheta(Theta(:,k));
    
    num = num + dlogPidtheta(:,k).^2 * J(k);
    den = den + dlogPidtheta(:,k).^2;
    
end

b = num ./ den;
b(isnan(b)) = 0;
% b = mean_J;

% Estimate gradient and Fisher information matrix
grad = 0;
F = 0;
for k = 1 : n_episodes
    grad = grad + dlogPidtheta(:,k) .* (J(k) - b);
    F = F + dlogPidtheta(:,k) * dlogPidtheta(:,k)';
end
grad = grad / n_episodes;
F = F / n_episodes;

% If we can compute the FIM in closed form, use it
if ismethod(pol_high,'fisher')
    F = pol_high.fisher;
end

% If we can compute the FIM inverse in closed form, use it
if ismethod(pol_high,'inverseFisher')
    invF = pol_high.inverseFisher;
    lambda = sqrt(grad' * (invF * grad) / (4 * lrate));
    lambda = max(lambda,1e-8); % to avoid numerical problems
    nat_grad = invF * grad / (2 * lambda);
elseif rank(F) == size(F,1)
    lambda = sqrt(grad' * (F \ grad) / (4 * lrate));
    lambda = max(lambda,1e-8);
    nat_grad = F \ grad / (2 * lambda);
else
    str = sprintf('WARNING: F is lower rank (rank = %d)!!! Should be %d', rank(F), size(F,1));
    disp(str);
    lambda = sqrt(grad' * (pinv(F) * grad) / (4 * lrate));
    lambda = max(lambda,1e-8);
    nat_grad = pinv(F) * grad / (2 * lambda);
end

end