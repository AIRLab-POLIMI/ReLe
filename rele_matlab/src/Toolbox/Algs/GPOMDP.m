%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dJdtheta, stepsize] = GPOMDP(policy, data, gamma, robj, lrate)

dlogpi_r = policy.dlogPidtheta;
dlogpi_c = 1;
dJdtheta = zeros(dlogpi_r, dlogpi_c);

num_trials = max(size(data));
parfor trial = 1 : num_trials
	sumdlogPi = zeros(dlogpi_r,dlogpi_c);
	for step = 1 : size(data(trial).a,2)
		sumdlogPi = sumdlogPi + ...
			policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        rew = gamma^(step-1) * data(trial).r(robj,step);
		dJdtheta = dJdtheta + sumdlogPi * rew;

	end
end

dJdtheta = dJdtheta / num_trials;

if nargin >= 5
    T = eye(length(dJdtheta)); % trasformation in Euclidean space
    lambda = sqrt(dJdtheta' * T * dJdtheta / (4 * lrate));
    lambda = max(lambda,1e-8); % to avoid numerical problems
    stepsize = 1 / (2 * lambda);
end