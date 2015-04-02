function dJdtheta = eREINFORCEbase(policy, data, gamma, robj)

dim   = length(policy.weights);
sdim  = length(data(1).x(1,:));
sigma = policy.stddev;
wnum  = policy.weights;

syms a
w      = sym('w',   [dim,  1]);
s      = sym('s',   [sdim, 1]);
phi    = [s(end:-1:1)];

pol = 1/(sqrt(2*pi) * (sigma)) * exp(-0.5*(a-w'*phi)^2/(sigma)^2);
% pretty(pol)
% eval(subs(pol, [w; k; phi; a], [wnum; knum; state; action]))

g = gradient(log(pol), w);

%%
% Reference:
% Peters, J. & Schaal, S.
% Reinforcement learning of motor skills with policy gradients
% Neural Networks, 2008, 21, 682-697

dlogpi_r = length(policy.weights);
dlogpi_c = 1;
dJdtheta = zeros(dlogpi_r, dlogpi_c);

%%% Compute optimal baseline
j=0; num_trials = max(size(data));
bnum = zeros(dlogpi_r,1); bden = zeros(dlogpi_r,1);
for trial = 1 : num_trials
    sumrew = 0;
    sumdlogPi = zeros(dlogpi_r,dlogpi_c);
    
    for step = 1 : max(size(data(trial).u))-1
        
        evalg = eval(subs(g, [w; s; a], [wnum; data(trial).x(step,:)'; data(trial).u(step,:)']));
        %         evalg'
        sumdlogPi = sumdlogPi + evalg;
        sumrew = sumrew + gamma^(step-1) * data(trial).r(step,robj);
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
    %     df = 1;
    
    for step = 1 : max(size(data(trial).u))-1
        evalg = eval(subs(g, [w; s; a], [wnum; data(trial).x(step,:)'; data(trial).u(step,:)']));
        %         evalg'
        sumdlogPi = sumdlogPi + evalg;
        
        sumrew = sumrew + gamma^(step-1) * data(trial).r(step,robj);
        %         sumrew = sumrew + df * data(trial).r(step);
        
        %         df = df * gamma;
        j = j + 1; % number of steps
    end
    dJdtheta = dJdtheta + sumdlogPi .* (ones(dlogpi_r, 1) * sumrew-b);
end

if gamma == 1
    dJdtheta = dJdtheta / j;
else
    % 	dJdtheta = (1 - gamma) * dJdtheta / num_trials;
    dJdtheta = dJdtheta / num_trials
end
