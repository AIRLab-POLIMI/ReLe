function grad = PGPEbase (pol_high, J, Theta)

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

% Estimate gradient
grad = 0;
for k = 1 : n_episodes
    grad = grad + dlogPidtheta(:,k) .* (J(k) - b);
end
grad = grad / n_episodes;

end
