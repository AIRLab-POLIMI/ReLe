clear all
domain = 'dam';
robj = 1;
[n_obj, pol, episodes, steps, gamma, is_avg, max_obj] = settings(domain);
iter = 0;
theta = [];

% Settings
tolerance = 0.1;
Hmin = -20; % with Gaussian policies the (differential) entropy can be negative
lrate = 4;

%% Learning
while true
    
    iter = iter + 1;
    [ds, uJ, dJ, H] = collect_samples(domain,episodes,steps,pol,is_avg,gamma);
    if gamma == 1
        J = uJ;
    else
        J = dJ;
    end
    
%     grad = finite_difference(pol,episodes,J,domain,robj);
%     grad = GPOMDPbase(pol,ds,gamma,robj);
%     grad = eREINFORCEbase(pol,ds,gamma,robj);
%     grad = eNACbase(pol,ds,gamma,robj);
    grad = eNAC(pol,ds,gamma,robj);
    
    theta = [theta; pol.theta'];
    norm_g = norm(grad);
    
    str_obj = strtrim(sprintf('%.4f, ', (J .* max_obj)));
    str_obj(end) = [];
    fprintf('%d ) H = %.2f, dev = %.4f, J = [ %s ]\n', iter, H, norm_g, str_obj)
    
    if norm_g < tolerance || H < Hmin
        break
    end
    
    grad_n = grad / max(norm(grad),1);
    pol = pol.update(lrate * grad_n);

end