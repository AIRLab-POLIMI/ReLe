%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dJdtheta = GPOMDP_IRL(policy, data, gamma, fReward)

dlogpi_r = policy.dlogPidtheta;
dlogpi_c = 1;
dJdtheta = 0;

totstep = 0;

num_trials = max(size(data));
for trial = 1 : num_trials
    sumdlogPi = zeros(dlogpi_r,dlogpi_c);
    for step = 1 : max(size(data(trial).a))
        sumdlogPi = sumdlogPi + ...
            policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        rew = gamma^(step-1) * fReward(data(trial).s(:,step), data(trial).a(:,step), data(trial).nexts(:,step));
        dJdtheta = dJdtheta + sumdlogPi * rew;
        totstep = totstep + 1;
    end
end

if gamma == 1
    dJdtheta = dJdtheta / totstep;
else
    dJdtheta = dJdtheta / num_trials;
end

end
