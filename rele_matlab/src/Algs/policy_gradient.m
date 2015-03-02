clear all
domain = 'deep';
robj = 1;
[n_obj, pol, episodes, steps, gamma, avg_rew_setting, max_obj] = settings(domain);
i = 0;
theta = [];

% Settings
tolerance = 0.1;
Hmin = -20; % with Gaussian policies the (differential) entropy can be negative
lrate = .1;

while true
    
    i = i + 1;
    [ds, uJ, dJ, H] = collect_samples(domain,episodes,steps,pol,avg_rew_setting,gamma);
    if gamma == 1
        J = uJ;
    else
        J = dJ;
    end
    
%     grad = finite_difference(pol,100,J,domain,robj);
%     grad = GPOMDPbase(pol,ds,gamma,robj);
%     grad = eREINFORCEbase(pol,ds,gamma,robj);
%     grad = eNACbase(pol,ds,gamma,robj);
    grad = eNAC(pol,ds,gamma,robj);
    
    theta = [theta; pol.theta'];
    norm_g = norm(grad);
    
    str_obj = strtrim(sprintf('%g, ', (J .* max_obj)));
    str_obj(end) = [];
    fprintf('%g ) H = %g, \t dev = %g, \t J = [ %s ]\n', i, H, norm_g, str_obj)
    
    if norm_g < tolerance || H < Hmin
        break
    end
    
    grad_n = grad / max(norm(grad),1);
    pol = pol.update(lrate * grad_n);

end
