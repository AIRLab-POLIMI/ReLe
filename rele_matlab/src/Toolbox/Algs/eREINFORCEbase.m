%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dJdtheta = eREINFORCEbase(policy, data, gamma, robj)

dlogpi_r = policy.dlogPidtheta;
dlogpi_c = 1;
dJdtheta = zeros(dlogpi_r, dlogpi_c);

%%% Compute optimal baseline
j = 0;
num_trials = max(size(data));
bnum = zeros(dlogpi_r,1);
bden = zeros(dlogpi_r,1);
for trial = 1 : num_trials
    sumrew = 0;
    sumdlogPi = zeros(dlogpi_r,dlogpi_c);
    
    for step = 1 : max(size(data(trial).a))
        sumdlogPi = sumdlogPi + ...
            policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        sumrew = sumrew + gamma^(step-1) * data(trial).r(robj,step);
        j = j + 1;
    end
    
    sumdlogPi = sumdlogPi .* sumdlogPi;
    bnum = bnum + sumdlogPi * sumrew;
    bden = bden + sumdlogPi;
end
b = bnum ./ bden;

%%% Compute gradient
j = 0;
for trial = 1 : num_trials
    sumrew = 0;
    sumdlogPi = zeros(dlogpi_r,dlogpi_c);
    
    for step = 1 : max(size(data(trial).a))
        sumdlogPi = sumdlogPi + ...
            policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        sumrew = sumrew + gamma^(step-1) * data(trial).r(robj, step);
        
        j = j + 1; % number of steps
    end
    dJdtheta = dJdtheta + sumdlogPi .* (ones(dlogpi_r, 1) * sumrew - b);
end

if gamma == 1
    dJdtheta = dJdtheta / j;
else
    % dJdtheta = (1 - gamma) * dJdtheta / num_trials;
    dJdtheta = dJdtheta / num_trials;
end
