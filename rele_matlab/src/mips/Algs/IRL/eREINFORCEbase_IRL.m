%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dJdtheta = eREINFORCEbase_IRL(policy, data, gamma, fReward)

dJdtheta = 0;

%%% Compute optimal baseline
j = 0;
num_trials = max(size(data));
bnum = 0;
bden = 0;
parfor trial = 1 : num_trials
    sumrew = 0;
    sumdlogPi = 0;
    
    for step = 1 : max(size(data(trial).a))
        sumdlogPi = sumdlogPi + ...
            policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        sumrew = sumrew + gamma^(step-1) * fReward(data(trial).s(:,step), data(trial).a(:,step),data(trial).nexts(:,step));
        j = j + 1;
    end
    
    sumdlogPi = sumdlogPi .* sumdlogPi;
    bnum = bnum + sumdlogPi * sumrew;
    bden = bden + sumdlogPi;
end
b = bnum ./ bden;
b(isnan(b)) = 0;

%%% Compute gradient
j = 0;
parfor trial = 1 : num_trials
    sumrew = 0;
    sumdlogPi = 0;
    
    for step = 1 : max(size(data(trial).a))
        sumdlogPi = sumdlogPi + ...
            policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        sumrew = sumrew + gamma^(step-1) * fReward(data(trial).s(:,step), data(trial).a(:,step),data(trial).nexts(:,step));
        
        j = j + 1; % number of steps
    end
    dJdtheta = dJdtheta + sumdlogPi .* (ones(size(b,1), 1) * sumrew - b);
end


dJdtheta = dJdtheta / num_trials;
