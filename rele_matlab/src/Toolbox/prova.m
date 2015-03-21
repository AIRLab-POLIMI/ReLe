function J = prova(point)

domain='deep';
maxepisodes = 1;
[n_obj, pol_low, ~, steps, gamma, is_avg, max_obj] = settings(domain);

J = zeros(maxepisodes,n_obj);
dim_theta = length(point);
Theta = zeros(dim_theta,maxepisodes);

for k = 1 : maxepisodes
    
    % Draw theta from the high-level policy and perform a rollout
    pol_tmp = pol_low;
    pol_tmp = pol_tmp.makeDeterministic;
    theta = point;
    pol_tmp.theta(1:dim_theta) = theta;
    Theta(:,k) = theta;
    [~, uJ, dJ] = collect_samples(domain,1,steps,pol_tmp,is_avg,gamma);
    if gamma == 1
        J(k,:) = uJ .* max_obj;
    else
        J(k,:) = dJ .* max_obj;
    end
    
end

J = J(:,1);
