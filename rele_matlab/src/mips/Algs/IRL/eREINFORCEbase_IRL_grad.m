%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dJdtheta, drewdJ] = eREINFORCEbase_IRL_grad(policy, data, gamma, fReward, dfReward)

dlogpi_r = policy.dlogPidtheta;
dlogpi_c = 1;
dJdtheta = 0;
drewdJ   = 0;

%%% Compute optimal baseline
j = 0;
num_trials = max(size(data));
bnum2 = 0;
bnum = 0;
bden = 0;
for trial = 1 : num_trials
    sumrew = 0;
    sumrew2 = 0;
    sumdlogPi = zeros(dlogpi_r,dlogpi_c);
    
    for step = 1 : max(size(data(trial).a))
        sumdlogPi = sumdlogPi + ...
            policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        sumrew = sumrew + gamma^(step-1) * fReward(data(trial).s(:,step), data(trial).a(:,step), data(trial).nexts(:,step));
        sumrew2 = sumrew2 + gamma^(step-1) * dfReward(data(trial).s(:,step), data(trial).a(:,step), data(trial).nexts(:,step));
        j = j + 1;
    end
    
    sumdlogPi = sumdlogPi .* sumdlogPi;
    bnum = bnum + sumdlogPi * sumrew;
    bnum2 = bnum2 + sumdlogPi * sumrew2;
    bden = bden + sumdlogPi;
end
b  = bnum  ./ bden;
b2 = bnum2 ./ repmat(bden,1,size(bnum2,2));
b(isnan(b))  = 0;
b2(isnan(b)) = 0;

%%% Compute gradient
j = 0;
for trial = 1 : num_trials
    sumrew = 0;
    sumrew2 = 0;
    sumdlogPi = zeros(dlogpi_r,dlogpi_c);
    
    for step = 1 : max(size(data(trial).a))
        sumdlogPi = sumdlogPi + ...
            policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        sumrew = sumrew + gamma^(step-1) * fReward(data(trial).s(:,step), data(trial).a(:,step), data(trial).nexts(:,step));
        sumrew2 = sumrew2 + gamma^(step-1) * dfReward(data(trial).s(:,step), data(trial).a(:,step), data(trial).nexts(:,step));
        
        j = j + 1; % number of steps
    end
    dJdtheta = dJdtheta + sumdlogPi .* (ones(dlogpi_r, 1) * sumrew - b);
    drewdJ = drewdJ + repmat(sumdlogPi,1,size(sumrew2,2)) .* (repmat(sumrew2,size(b2,1),1) - b2);
end

dJdtheta = dJdtheta / num_trials;
drewdJ = drewdJ / num_trials;
