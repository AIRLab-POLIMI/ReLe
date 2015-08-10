%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dJdtheta, drewdJ] = eREINFORCEbase_IRL_grad(policy, data, gamma, fReward, dfReward)

dlogpi_r = policy.dlogPidtheta;
dlogpi_c = 1;
dJdtheta = 0;
drewdJ   = 0;

%% Compute optimal baseline
bnum_rewder = 0;
bnum_rewfun = 0;
bden = 0;
num_trials = max(size(data));
parfor trial = 1 : num_trials
    sum_rewfun = 0;
    sum_rewder = 0;
    sumdlogPi = zeros(dlogpi_r,dlogpi_c);
    
    df = 1;
    
    for step = 1 : max(size(data(trial).a))
        sumdlogPi = sumdlogPi + ...
            policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        % sum discounted reward
        sum_rewfun = sum_rewfun + df * fReward(data(trial).s(:,step), data(trial).a(:,step), data(trial).nexts(:,step));
        % sum discounted reward derivative
        sum_rewder = sum_rewder + df * dfReward(data(trial).s(:,step), data(trial).a(:,step), data(trial).nexts(:,step));
        % update discount factor according to time step
        df = df * gamma;
    end
    
    sumdlogPi = sumdlogPi .* sumdlogPi;
    bnum_rewfun = bnum_rewfun + sumdlogPi * sum_rewfun;
    bnum_rewder = bnum_rewder + sumdlogPi * sum_rewder;
    bden = bden + sumdlogPi;
end
b  = bnum_rewfun ./ bden;
b2 = bnum_rewder ./ repmat(bden,1,size(bnum_rewder,2));
b(isnan(b))  = 0;
b2(isnan(b)) = 0;

%% Compute gradient
[nr,nc] = size(b2);
totstep = 0;
parfor trial = 1 : num_trials
    sum_rewfun = 0;
    sum_rewder = 0;
    sumdlogPi = zeros(dlogpi_r,dlogpi_c);
    
    df = 1.0;
    
    for step = 1 : max(size(data(trial).a))
        sumdlogPi = sumdlogPi + ...
            policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        sum_rewfun = sum_rewfun + df * fReward(data(trial).s(:,step), data(trial).a(:,step), data(trial).nexts(:,step));
        sum_rewder = sum_rewder + df * dfReward(data(trial).s(:,step), data(trial).a(:,step), data(trial).nexts(:,step));
        % update discount factor according to time step
        df = df * gamma;
        % count steps
        totstep = totstep + 1;
    end
    dJdtheta = dJdtheta + sumdlogPi .* (ones(dlogpi_r, 1) * sum_rewfun - b);
    drewdJ = drewdJ + repmat(sumdlogPi,1,nc) .* (repmat(sum_rewder,nr,1) - b2);
end

if gamma == 1
    dJdtheta = dJdtheta / totstep;
    drewdJ = drewdJ / totstep;
else
    dJdtheta = dJdtheta / num_trials;
    drewdJ = drewdJ / num_trials;
end

