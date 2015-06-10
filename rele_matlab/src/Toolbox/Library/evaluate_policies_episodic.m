function J = evaluate_policies_episodic ( policies, domain, makeDet )
% Given a set of high-level POLICIES, it returns the corresponding return J
% in the objectives space.
% Set MAKEDET to 1 if you want to make the policies deterministic.

[n_obj, ~, episodes] = settings(domain);

mdpconfig = [domain '_mdpvariables'];
mdp_vars = feval(mdpconfig);
isStochastic = mdp_vars.isStochastic;
if makeDet && ~isStochastic
    episodes = 1;
end

N_pol = numel(policies);
J = zeros(N_pol, n_obj);

parfor i = 1 : N_pol

%     fprintf('Evaluating policy %d of %d ...\n', i, N_pol)
    
    if makeDet
        policy = policies(i).makeDeterministic;
    else
        policy = policies(i);
    end
    
    J_ep = collect_episodes(domain, episodes, policy);
    J(i,:) = mean(J_ep,1);

end

end
