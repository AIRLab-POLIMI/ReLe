function h = HessianRF(policy_logdif,policy_logdif2, data, gamma, robj)

h = 0;
num_trials = max(size(data));
for trial = 1 : num_trials
    sumrew = 0;
    sumhlogPi = 0;
    sumdlogPi = 0;
    for step = 1 : max(size(data(trial).a))
        dlogpidtheta = policy_logdif(data(trial).s(:,step), data(trial).a(:,step));
        sumdlogPi = sumdlogPi + dlogpidtheta;
        hlogpidtheta = policy_logdif2(data(trial).s(:,step), data(trial).a(:,step));
        sumhlogPi = sumhlogPi + hlogpidtheta;
        sumrew = sumrew + gamma^(step-1) * data(trial).r(robj,step);
    end
    h = h + sumrew * (sumdlogPi * sumdlogPi' + sumhlogPi);
end

h = h / num_trials;

end