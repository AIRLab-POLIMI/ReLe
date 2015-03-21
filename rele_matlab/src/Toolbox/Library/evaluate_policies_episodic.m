% Given a set of high-level POLICIES, it returns the corresponding
% SOLUTIONS in the objectives space.
function solutions = evaluate_policies_episodic ( policies, domain )

[n_obj, pol_low, episodes, steps, gamma, is_avg, max_obj] = settings(domain);
pol_low = pol_low.makeDeterministic;

solutions = zeros(numel(policies), n_obj);
N_pol = numel(policies);

parfor i = 1 : N_pol
    
    theta = policies(i).makeDeterministic.drawAction;
    fprintf('Evaluating policy %d of %d ...\n', i, N_pol)
    dim_theta = policies(i).dim;
    pol_tmp = pol_low;
    pol_tmp.theta(1:dim_theta) = theta;
    
    [~, uJ, dJ] = collect_samples(domain, episodes, steps, pol_tmp, is_avg, gamma);
    
    if gamma == 1
        J = uJ;
    else
        J = dJ;
    end
    
    solutions(i,:) = (J .* max_obj)';

end

end
