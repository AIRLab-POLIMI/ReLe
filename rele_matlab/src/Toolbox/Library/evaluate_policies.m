% Given a set of low-level POLICIES, it returns the corresponding SOLUTIONS 
% in the objectives space.
function solutions = evaluate_policies ( policies, domain )

[n_obj, ~, episodes, steps, gamma, is_avg, max_obj] = settings(domain);

solutions = zeros(numel(policies), n_obj);
N_pol = numel(policies);

parfor i = 1 : N_pol

    fprintf('Evaluating policy %d of %d ...\n', i, N_pol)
    [~, uJ, dJ] = collect_samples(domain, episodes, steps, ...
        policies(i).makeDeterministic, is_avg, gamma);

    if gamma == 1
        J = uJ;
    else
        J = dJ;
    end
    
    solutions(i,:) = (J .* max_obj)';
    
end

end
