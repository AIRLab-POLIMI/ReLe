function [J, Theta] = collect_episodes(domain, maxepisodes, pol_high)
% Collects episodes for the specified domain. The low-level policy is
% deterministic and its parameters are drawn from a high-level
% distribution (pol_high).

[n_obj, pol_low, ~, steps] = settings(domain);
pol_low = pol_low.makeDeterministic;

J = zeros(maxepisodes,n_obj);
dim_theta = pol_high.dim;
Theta = zeros(dim_theta,maxepisodes);

%%%%%%%%%%%%%%%%%%%
[mexParams, gamma, max_obj] = getMexParams(domain);
%%%%%%%%%%%%%%%%%%%

parfor k = 1 : maxepisodes
    
    % Draw theta from the high-level policy and perform a rollout
    pol_tmp = pol_low;
    theta = pol_high.drawAction;
    pol_tmp.theta(1:dim_theta) = theta;
    Theta(:,k) = theta;

%     [~, J_ep] = collect_samples(domain, 1, steps, pol_tmp);

%%%%%%%%%%%%%%%%%%%
    mm = mexParams;
    mm.policyParameters = pol_tmp.theta;
    [~, J_ep] = collectSamples(domain, 1, steps, gamma, mm);
    J_ep = mean(J_ep,1) .* abs(max_obj)';
%%%%%%%%%%%%%%%%%%%
    
    J(k,:) = J_ep;
    
end

end
