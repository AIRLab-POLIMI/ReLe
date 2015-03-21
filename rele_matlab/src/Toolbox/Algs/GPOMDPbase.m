%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dJdtheta = GPOMDPbase(policy, data, gamma, robj)

dlogpi_r = policy.dlogPidtheta;
dlogpi_c = 1;
dJdtheta = zeros(dlogpi_r, dlogpi_c);

%%% Compute baselines
num_trials = max(size(data));
bnum = zeros(dlogpi_r, max(size(data(1).a)));
bden = zeros(dlogpi_r, max(size(data(1).a)));
for trial = 1 : num_trials
	sumdlogPi = zeros(dlogpi_r,dlogpi_c);
    num_steps = max(size(data(trial).a));

    if size(bnum,2) < num_steps
        dif = num_steps - size(bnum,2);
        bnum = [bnum, zeros(dlogpi_r, dif)];
        bden = [bden, zeros(dlogpi_r, dif)];
    end
    
	for step = 1 : num_steps
		sumdlogPi = sumdlogPi + ...
			policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
		rew = gamma^(step - 1) * data(trial).r(robj,step);
		sumdlogPi2 = sumdlogPi .* sumdlogPi;
		bnum(:,step) = bnum(:, step) + sumdlogPi2 * rew;
		bden(:,step) = bden(:, step) + sumdlogPi2; 
	end
end

b = bnum ./ bden;

%%% Compute gradient
j = 0;
for trial = 1 : num_trials
	sumdlogPi = zeros(dlogpi_r,dlogpi_c);
	for step = 1 : max(size(data(trial).a)) 
        sumdlogPi = sumdlogPi + ...
			policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        rew = gamma^(step-1) * data(trial).r(robj,step);
		dJdtheta = dJdtheta + sumdlogPi .* (ones(dlogpi_r, 1) * rew - b(:,step));

        j = j + 1; % number of steps
	end
end

if gamma == 1
	dJdtheta = dJdtheta / j;
else
    % dJdtheta = (1 - gamma) * dJdtheta / num_trials;
	dJdtheta = dJdtheta / num_trials;
end
