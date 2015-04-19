%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dJdtheta, stepsize] = eREINFORCEbase(policy_logdif, data, gamma, robj, lrate)

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
            policy_logdif(data(trial).s(:,step), data(trial).a(:,step));
        sumrew = sumrew + gamma^(step-1) * data(trial).r(robj,step);
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
            policy_logdif(data(trial).s(:,step), data(trial).a(:,step));
        sumrew = sumrew + gamma^(step-1) * data(trial).r(robj, step);
        
        j = j + 1; % number of steps
    end
    dJdtheta = dJdtheta + sumdlogPi .* (ones(size(b,1), 1) * sumrew - b);
end

dJdtheta = dJdtheta / num_trials;

if nargin >= 5
    T = eye(length(dJdtheta)); % trasformation in Euclidean space
    lambda = sqrt(dJdtheta' * T * dJdtheta / (4 * lrate));
    lambda = max(lambda,1e-8); % to avoid numerical problems
    stepsize = 1 / (2 * lambda);
end
