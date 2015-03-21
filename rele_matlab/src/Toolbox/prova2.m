f = @prova;
domain='deep';
[n_obj, pol_low, ~, steps, gamma, is_avg, max_obj] = settings(domain);
d = length(pol_low.theta);
[xopt,fopt] = xnes(f, d, zeros(d,1), 40)