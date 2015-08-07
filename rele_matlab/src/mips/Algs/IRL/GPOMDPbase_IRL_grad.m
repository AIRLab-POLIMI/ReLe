%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dJdtheta, drewdJ] = GPOMDPbase_IRL_grad(policy, data, gamma, fReward, dfReward)

dlogpi_r = policy.dlogPidtheta;
dlogpi_c = 1;
dJdtheta = zeros(dlogpi_r, dlogpi_c);
drewdJ = 0;

%%% Compute baselines
num_trials = max(size(data));
bnum = zeros(dlogpi_r, max(size(data(1).a)));
bnum2 = cell(1, max(size(data(1).a)));
for i = 1:length(bnum2)
    bnum2{i} = 0;
end
bden = zeros(dlogpi_r, max(size(data(1).a)));

for trial = 1 : num_trials
	sumdlogPi = zeros(dlogpi_r,dlogpi_c);
    num_steps = max(size(data(trial).a));

    if size(bnum,2) < num_steps
        dif = num_steps - size(bnum,2);
        bnum = [bnum, zeros(dlogpi_r, dif)];
        bden = [bden, zeros(dlogpi_r, dif)];
        dd = length(bnum2);
        for ppp = 1 : dif
            bnum2{dd+ppp} = 0;
        end
    end
    
	for step = 1 : num_steps
		sumdlogPi = sumdlogPi + ...
			policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
		rew  = gamma^(step - 1) * fReward(data(trial).s(:,step), data(trial).a(:,step), data(trial).nexts(:,step));
        rew2 = gamma^(step - 1) * dfReward(data(trial).s(:,step), data(trial).a(:,step), data(trial).nexts(:,step));
		sumdlogPi2 = sumdlogPi .* sumdlogPi;
		bnum(:,step) = bnum(:, step) + sumdlogPi2 * rew;
        bnum2{step}  = bnum2{step} + sumdlogPi2 * rew2;
		bden(:,step) = bden(:, step) + sumdlogPi2; 
	end
end

b = bnum ./ bden;
b(isnan(b)) = 0;

for i = 1:length(bnum2)
    tmp =  bnum2{i} ./ repmat(bden(:,i),1,size(bnum2{i},2));
    tmp(isnan(tmp)) = 0;
    b2{i} = tmp;
end


%%% Compute gradient
j = 0;
for trial = 1 : num_trials
	sumdlogPi = zeros(dlogpi_r,dlogpi_c);
	for step = 1 : max(size(data(trial).a)) 
        sumdlogPi = sumdlogPi + ...
			policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        rew = gamma^(step-1) * fReward(data(trial).s(:,step), data(trial).a(:,step), data(trial).nexts(:,step));
        rew2 = gamma^(step - 1) * dfReward(data(trial).s(:,step), data(trial).a(:,step), data(trial).nexts(:,step));
		dJdtheta = dJdtheta + sumdlogPi .* (ones(dlogpi_r, 1) * rew - b(:,step));
        drewdJ   = drewdJ + repmat(sumdlogPi,1,size(rew2,2)) .* (repmat(rew2,size(b2{step},1),1) - b2{step});

        j = j + 1; % number of steps
	end
end

	dJdtheta = dJdtheta / num_trials;

