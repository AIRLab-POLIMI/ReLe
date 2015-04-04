function dJdtheta = gGPOMDPbase(policy, data, gamma, robj)

disp('GPOMDP with baseline');

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

%%% Compute baselines
num_trials = max(size(data));
bnum = zeros(dlogpi_r, max(size(data(1).u)));
bden = zeros(dlogpi_r, max(size(data(1).u)));
for trial = 1 : num_trials
    sumdlogPi = zeros(dlogpi_r,dlogpi_c);
    csteps = max(size(data(trial).u))-1;
    if size(bnum,2) < csteps
        dif = csteps-size(bnum,2);
        bnum = [bnum, zeros(dlogpi_r, dif)];
        bden = [bden, zeros(dlogpi_r, dif)];
    end
    for step = 1 : csteps
        evalg = policyg.dlogPidtheta(data(trial).x(step,:)', 1+data(trial).u(step,:)');
        %         evalg'
        sumdlogPi = sumdlogPi + evalg;
        
        rew = gamma^(step - 1) * data(trial).r(step,robj);
        sumdlogPi2 = sumdlogPi .* sumdlogPi;
        bnum(:,step) = bnum(:, step) + sumdlogPi2 * rew;
        bden(:,step) = bden(:, step) + sumdlogPi2;
    end
end
b = zeros(size(bnum));
b(bden~=0) = bnum(bden~=0) ./ bden(bden~=0);

%%% Compute gradient
j = 0;
for trial = 1 : num_trials
    % 	df = 1;
    sumdlogPi = zeros(dlogpi_r,dlogpi_c);
    for step = 1 : max(size(data(trial).u))-1
        
        %         disp([data(trial).s(:,step)',data(trial).a(:,step)'])
        %         disp('log')
        %         			disp(policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(step))')
        %                     disp('feat')
        %                     disp(feval(@dam_basis_rbf, data(trial).s(:,step))')
        evalg = policyg.dlogPidtheta(data(trial).x(step,:)', 1+data(trial).u(step,:)');
        %         evalg'
        sumdlogPi = sumdlogPi + evalg;
        
        %          disp('sumdlogPi')
        %         			disp(sumdlogPi')
        
        % 		rew = df * data(trial).r(step);
        rew = gamma^(step-1) * data(trial).r(step,robj);
        %          disp('rew')
        %         			disp(rew)
        
        dJdtheta = dJdtheta + sumdlogPi .* (ones(dlogpi_r, 1) * rew - b(:,step));
        %          disp('dJdtheta')
        %         			disp(dJdtheta')
        
        % 		df = df * gamma;
        j = j + 1; % number of steps
    end
end

if gamma == 1
    dJdtheta = dJdtheta / num_trials;
else
    % 	dJdtheta = (1 - gamma) * dJdtheta / num_trials;
    dJdtheta = dJdtheta / num_trials
end
