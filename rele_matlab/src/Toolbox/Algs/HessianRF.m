function HJ = HessianRF(policy, data, gamma, robj)

dlp = policy.dlogPidtheta;
HJ = zeros(dlp, dlp);

j = 0;
num_trials = max(size(data));
for trial = 1 : num_trials
    sumrew = 0;
    sumhlogPi = zeros(dlp,dlp);
    sumdlogPi = zeros(dlp,1);
    for step = 1 : max(size(data(trial).a))
        dlogpidtheta = policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        sumdlogPi = sumdlogPi + dlogpidtheta;
        hlogpidtheta = policy.hlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        sumhlogPi = sumhlogPi + hlogpidtheta;
        sumrew = sumrew + gamma^(step-1) * data(trial).r(robj,step);
        
        j = j + 1; % number of steps
    end
    HJ = HJ + sumrew * (sumdlogPi * sumdlogPi' + sumhlogPi);
end

if gamma == 1
    HJ = HJ / j;
else
%     dJdtheta = (1 - gamma) * dJdtheta / num_trials;
    HJ = HJ / num_trials;
end

end