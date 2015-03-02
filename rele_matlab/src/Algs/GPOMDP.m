%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dJdtheta = GPOMDP(policy, data, gamma, robj)

dlogpi_r = policy.dlogPidtheta;
dlogpi_c = 1;
dJdtheta = zeros(dlogpi_r, dlogpi_c);

j = 0;
num_trials = max(size(data));
for trial = 1 : num_trials
	sumdlogPi = zeros(dlogpi_r,dlogpi_c);
	for step = 1 : max(size(data(trial).a)) 
		sumdlogPi = sumdlogPi + ...
			policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        rew = gamma^(step-1) * data(trial).r(robj,step);
		dJdtheta = dJdtheta + sumdlogPi * rew;

        j = j + 1; % number of steps
	end
end

if gamma == 1
	dJdtheta = dJdtheta / j;
else
%     dJdtheta = (1 - gamma) * dJdtheta / num_trials;
	dJdtheta = dJdtheta / num_trials;
end
