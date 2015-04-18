%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dJdtheta, stepsize] = GPOMDPbase(policy_logdif, dimp, data, gamma, robj, lrate)
dJdtheta = 0;

%%% Compute baselines
num_trials = max(size(data));

actions = cell(1,numel(data));
[actions{:}] = data.a;
lengths = cellfun('length',actions);

bnum = zeros(dimp, max(lengths));
bden = zeros(dimp, max(lengths));
for trial = 1 : num_trials
	sumdlogPi = 0;
    num_steps = max(size(data(trial).a));

	for step = 1 : num_steps
		sumdlogPi = sumdlogPi + ...
			policy_logdif(data(trial).s(:,step), data(trial).a(:,step));
		rew = gamma^(step - 1) * data(trial).r(robj,step);
		sumdlogPi2 = sumdlogPi .* sumdlogPi;
		bnum(:,step) = bnum(:, step) + sumdlogPi2 * rew;
		bden(:,step) = bden(:, step) + sumdlogPi2; 
	end
end

b = bnum ./ bden;
b(isnan(b)) = 0;

%%% Compute gradient
j = 0;
for trial = 1 : num_trials
	sumdlogPi = 0;
	for step = 1 : max(size(data(trial).a)) 
        sumdlogPi = sumdlogPi + ...
			policy_logdif(data(trial).s(:,step), data(trial).a(:,step));
        rew = gamma^(step-1) * data(trial).r(robj,step);
		dJdtheta = dJdtheta + sumdlogPi .* (ones(dimp, 1) * rew - b(:,step));

        j = j + 1; % number of steps
	end
end

dJdtheta = dJdtheta / num_trials;

if nargin >= 6
    T = eye(length(dJdtheta)); % trasformation in Euclidean space
    lambda = sqrt(dJdtheta' * T * dJdtheta / (4 * lrate));
    lambda = max(lambda,1e-8); % to avoid numerical problems
    stepsize = 1 / (2 * lambda);
end
