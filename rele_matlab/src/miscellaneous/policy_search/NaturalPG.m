function [dJdtheta, stepsize] = NaturalPG(alg, policy_logdif, dlp, data, gamma, robj, lrate)

dlogpi_r = dlp;
fisher = zeros(dlogpi_r,dlogpi_r);

num_trials = max(size(data));
% j = 0;
parfor trial = 1 : num_trials
	for step = 1 : max(size(data(trial).a)) 
		loggrad = policy_logdif(data(trial).s(:,step), data(trial).a(:,step));
        fisher = fisher + loggrad * loggrad';
%         j = j + 1;
	end
end
fisher = fisher / num_trials;

if strcmp(alg, 'r') 
    grad = eREINFORCE(policy_logdif, data, gamma, robj);
elseif strcmp(alg, 'rb')
    grad = eREINFORCEbase(policy_logdif, data, gamma, robj);
elseif strcmp(alg, 'g')
    grad = GPOMDP(policy_logdif, data, gamma, robj);
elseif strcmp(alg, 'gb')
    grad = GPOMDPbase(policy_logdif, dlp, data, gamma, robj);
else
    error('Unknown algoritm');
end

if rank(fisher) == dlogpi_r
    dJdtheta = fisher \ grad;
else
	str = sprintf('WARNING: F is lower rank (rank = %d)!!! Should be %d', rank(fisher), dlogpi_r);
	disp(str);
    dJdtheta = pinv(fisher) * grad;
end
    
if nargin >= 7
    T = eye(length(dJdtheta)); % trasformation in Euclidean space
    lambda = sqrt(dJdtheta' * T * dJdtheta / (4 * lrate));
    lambda = max(lambda,1e-8); % to avoid numerical problems
    stepsize = 1 / (2 * lambda);
end