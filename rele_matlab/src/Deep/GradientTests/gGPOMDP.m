function dJdtheta = gGPOMDP(policy, data, gamma, obj)

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

j = 0; num_trials = max(size(data));
for trial = 1 : num_trials
% 	df = 1;
% disp('-------------');
	sumdlogPi = zeros(dlogpi_r,dlogpi_c);
	for step = 1 : max(size(data(trial).u))-1
        
        evalg = policyg.dlogPidtheta(data(trial).x(step,:)', 1+data(trial).u(step,:)');
        
%         evalg'
		sumdlogPi = sumdlogPi + evalg;
% 	        disp('sumdlogPi')
%         disp(sumdlogPi)
        
% 		rew = df * data(trial).r(step);
        rew = gamma^(step-1) * data(trial).r(step,obj);
        
%         disp('rew')
%         disp(rew)
        
		dJdtheta = dJdtheta + sumdlogPi * rew;

%                 disp('dJdtheta')
%         disp(dJdtheta)
        
% 		df = df * gamma;
		j = j + 1; % number of steps
	end
end

if gamma == 1
	dJdtheta = dJdtheta / num_trials;
else
%     dJdtheta = (1 - gamma) * dJdtheta / num_trials;
	dJdtheta = dJdtheta / num_trials
end
