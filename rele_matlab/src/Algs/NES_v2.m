%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NES with diagonal covariance matrix. It has a slightly faster computation 
% of the gradient than 'NES.m'.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear variables;
close all;

%% Initialize policy;
domain = 'deep';
robj = 1;
[~, pol_low, episodes, steps, gamma, avg_rew_setting, max_obj] = settings(domain);

N_samples   = 50;
N_params    = size(pol_low.theta,1);

% If the policy has a learnable variance, we don't want to learn it and
% we make it deterministic
N_params = N_params - pol_low.dim_variance_params;
pol_low = pol_low.makeDeterministic;

mu          = 0 * ones(1, N_params);
sigma       = eye ( N_params ) * 10; % change according to the domain

lrate       = 1;
maxIter     = 200;

meanReward  = zeros(maxIter, 1);

updateSigma = true;
sigma_min   = .1;

%% Policy Gradient
for j = 1 : maxIter
    
    assert(isrow(mu));
    
    rewards = zeros( N_samples, 1 );
    actionSamples = mvnrnd(mu, sigma.^2, N_samples);

    parfor i = 1 : N_samples
        pol_tmp = pol_low;
        pol_tmp.theta(1:N_params) = actionSamples(i,:)';
        [ds, uJ, dJ] = collect_samples(domain,episodes,steps,pol_tmp,avg_rew_setting,gamma);
        if gamma == 1
            rewards(i) = uJ(robj) .* max_obj(robj);
        else
            rewards(i) = dJ(robj) .* max_obj(robj);
        end
    end
    
    meanReward(j) = mean(rewards);
    
    A             = bsxfun( @minus, actionSamples, mu );
    dlogpi_mu     = bsxfun( @rdivide, A, diag(sigma)'.^2 );
    dlogpi_sigma  = bsxfun( @plus, -(diag(sigma)').^-1, ...
        bsxfun(@rdivide, A.^2, (diag(sigma)').^3) );

    dlogpi        = [dlogpi_mu, dlogpi_sigma];

    num           = sum( bsxfun(@times, dlogpi.^2, rewards) );
    den           = sum( dlogpi.^2 );
    baseline      = num ./ den;
%     baseline      = meanReward(j);
    
	B             = bsxfun(@minus, rewards, baseline);
    grad          = sum( bsxfun(@times, dlogpi, B) );
    
    %%% Natural gradient
    F = zeros(N_params*2);
    parfor k = 1 : N_samples
        F = F + dlogpi(k,:)' * dlogpi(k,:);
    end
    grad = grad / N_samples;
    F = F / N_samples;
    if rank(F) == size(F,1)
        nat_grad = F \ grad';
        lambda = sqrt(grad * (F \ grad') / (4 * lrate));
        lambda = max(lambda,1e-8); % to avoid numerical problems
    else
        str = sprintf('WARNING: F is lower rank (rank = %d)!!! Should be %d', rank(F), size(F,1));
        disp(str);
        lambda = sqrt(grad * (pinv(F) * grad') / (4 * lrate));
        lambda = max(lambda,1e-8); % to avoid numerical problems
        nat_grad = pinv(F) * grad' / (2 * lambda);
    end
    newParams{j} = [mu'; diag(sigma)] + nat_grad / (2 * lambda);
    %%%
    
    %%% Vanilla gradient
%     newParams{j} = [mu'; diag(sigma)] + learningRate * grad';
    %%%
    mu = newParams{j}(1:N_params)';
    
    if updateSigma
        t_sigma = newParams{j}(N_params+1:end);
        for i = 1 : N_params
           t_sigma(i) = max(sigma_min, t_sigma(i));
        end
        sigma = diag(t_sigma);
    end
        
    fprintf( 'Iteration %d, Mean Reward: %f\n', j, meanReward(j) );

end

%% Plot
plot(meanReward,'DisplayName',['lrate = ' num2str(lrate)])
legend('Location','BestOutside')
