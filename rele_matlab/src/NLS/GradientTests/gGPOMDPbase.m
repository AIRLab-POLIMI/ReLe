function dJdtheta = gGPOMDPbase(policy, data, gamma, robj)

disp('GPOMDP with baseline');

dim   = length(policy.weights);
sdim  = length(data(1).x(1,:));
sigma = policy.stddev;
wnum  = policy.weights;

syms a
w      = sym('w',   [dim,  1]);
s      = sym('s',   [sdim, 1]);
phi    = [s(end:-1:1)];

varsigma = sigma * sum(phi);
pol = 1/(sqrt(2*pi) * (varsigma)) * exp(-0.5*(a-w'*phi)^2/(varsigma)^2);
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
        evalg = eval(subs(g, [w; s; a], [wnum; data(trial).x(step,:)'; data(trial).u(step,:)']));
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
        evalg = eval(subs(g, [w; s; a], [wnum; data(trial).x(step,:)'; data(trial).u(step,:)']));
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
    dJdtheta = dJdtheta / j;
else
    % 	dJdtheta = (1 - gamma) * dJdtheta / num_trials;
    dJdtheta = dJdtheta / num_trials
end
