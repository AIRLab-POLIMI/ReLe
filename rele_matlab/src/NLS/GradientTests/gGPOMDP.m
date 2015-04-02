function dJdtheta = gGPOMDP(policy, data, gamma, obj)

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

% h = hessian(log(pol), w);
% evalh = eval(subs(h, [w; k; phi; a], [wnum; knum; state; action]));


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
        
        evalg = eval(subs(g, [w; s; a], [wnum; data(trial).x(step,:)'; data(trial).u(step,:)']));
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
	dJdtheta = dJdtheta / j;
else
%     dJdtheta = (1 - gamma) * dJdtheta / num_trials;
	dJdtheta = dJdtheta / num_trials
end
