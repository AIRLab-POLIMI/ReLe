% Given a set of POLICIES, it returns the corresponding SOLUTIONS in the
% objectives space.
function solutions = evaluate_policies ( policies, domain )

[N, ~, episodes, steps, gamma, avg_rew_setting, max_obj] = settings(domain);

solutions = zeros(numel(policies), N);
N_pol = numel(policies);
parfor i = 1 : N_pol

    fprintf('Evaluating policy %d of %d ...\n', i, N_pol)
    [~, uJ, dJ] = collect_samples(domain, episodes, steps, policies(i), ...
        avg_rew_setting, gamma);

    if gamma == 1
        J = uJ;
    else
        J = dJ;
    end
    
    solutions(i,:) = (J .* max_obj)';
    
end

end
