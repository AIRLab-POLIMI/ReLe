%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dJdtheta = GPOMDPbase_IRL(policy, data, gamma, fReward)

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
		rew = gamma^(step - 1) * fReward(data(trial).s(:,step), data(trial).a(:,step),data(trial).nexts(:,step));
		sumdlogPi2 = sumdlogPi .* sumdlogPi;
		bnum(:,step) = bnum(:, step) + sumdlogPi2 * rew;
		bden(:,step) = bden(:, step) + sumdlogPi2; 
	end
end

b = bnum ./ bden;
b(isnan(b)) = 0;

%%% Compute gradient
for trial = 1 : num_trials
	sumdlogPi = zeros(dlogpi_r,dlogpi_c);
	for step = 1 : max(size(data(trial).a)) 
        sumdlogPi = sumdlogPi + ...
			policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        rew = gamma^(step-1) * fReward(data(trial).s(:,step), data(trial).a(:,step),data(trial).nexts(:,step));
		dJdtheta = dJdtheta + sumdlogPi .* (ones(dlogpi_r, 1) * rew - b(:,step));
	end
end

dJdtheta = dJdtheta / num_trials;
