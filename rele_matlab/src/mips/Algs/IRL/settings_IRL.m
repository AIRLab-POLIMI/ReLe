function [ds, policy, gamma] = settings_IRL(domain)

if strcmp(domain, 'lqr') == 1
    [~, policy, episodes, steps, gamma] = lqr_settings();
    LQR = lqr_environment(1);
    val = dlqr(LQR.A, LQR.B, LQR.Q, LQR.R);
    policy.theta = -val;
    % policy = pol.makeDeterministic;    
    [ds, ~] = collect_samples(domain,episodes,steps,policy);
else
    error('Unknown domain');
end
end