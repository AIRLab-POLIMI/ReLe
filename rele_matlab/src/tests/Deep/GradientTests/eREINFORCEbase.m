function dJdtheta = eREINFORCEbase(policy, data, gamma, robj)

dim   = length(policy.weights);
sdim  = length(data(1).x(1,:));
it    = policy.it;
wnum  = policy.weights;

mdp_vars = deep_mdpvariables();
policyg = gibbs_policy(@deep_basis_pol_v1, ...
    zeros(deep_basis_pol_v1,1), ...
    mdp_vars.action_list);
policyg.theta = policy.weights;
policyg.inverse_temperature = it;

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
        
        evalg = policyg.dlogPidtheta(data(trial).x(step,:)', 1+data(trial).u(step,:)');
        %         evalg'
        sumdlogPi = sumdlogPi + evalg;
        sumrew = sumrew + gamma^(step-1) * data(trial).r(step,robj);
        j = j + 1;
    end
    sumdlogPi = sumdlogPi .* sumdlogPi;
    bnum = bnum + sumdlogPi * sumrew;
    bden = bden + sumdlogPi;
end
b = zeros(size(bnum));
b(bden~=0) = bnum(bden~=0) ./ bden(bden~=0);

%%% Compute gradient
j = 0;
for trial = 1 : num_trials
    sumrew = 0;
    sumdlogPi = zeros(dlogpi_r,dlogpi_c);
    %     df = 1;
    
    for step = 1 : max(size(data(trial).u))-1
        evalg = policyg.dlogPidtheta(data(trial).x(step,:)', 1+data(trial).u(step,:)');
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
    dJdtheta = dJdtheta / num_trials;
else
    % 	dJdtheta = (1 - gamma) * dJdtheta / num_trials;
    dJdtheta = dJdtheta / num_trials
end
