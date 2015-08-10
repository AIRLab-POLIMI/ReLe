%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dJdtheta = eREINFORCE_IRL(policy, data, gamma, fReward)

dlogpi_r = policy.dlogPidtheta;
dlogpi_c = 1;
dJdtheta = 0;

totstep = 0;

num_trials = max(size(data));
parfor trial = 1 : num_trials
    sumrew = 0;
    sumdlogPi = zeros(dlogpi_r,dlogpi_c);
    
    df = 1;
    
    for step = 1 : max(size(data(trial).a))
        sumdlogPi = sumdlogPi + ...
            policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        sumrew = sumrew + df * fReward(data(trial).s(:,step), data(trial).a(:,step),data(trial).nexts(:,step));
        
        totstep = totstep + 1;
        df = df * gamma;
    end
    dJdtheta = dJdtheta + sumdlogPi * sumrew;
end

if gamma == 1
    dJdtheta = dJdtheta / totstep;
else
    dJdtheta = dJdtheta / num_trials;
end

