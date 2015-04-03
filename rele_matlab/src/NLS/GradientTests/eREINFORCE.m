function dJdtheta = eREINFORCE(policy, data, gamma, obj, cppdata)

dim   = length(policy.weights);
sdim  = length(data(1).x(1,:));
sigma = policy.stddev;
wnum  = policy.weights;

syms a
w      = sym('w',   [dim,  1]);
s      = sym('s',   [sdim, 1]);
phi    = s(end:-1:1);

varsigma = sigma * sum(phi);
pol = 1/(sqrt(2*pi) * (varsigma)) * exp(-0.5*(a-w'*phi)^2/(varsigma)^2);
% pol = 1/(sqrt(2*pi) * (sigma)) * exp(-0.5*(a-w'*phi)^2/(sigma)^2);
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
    sumrew = 0;
    sumdlogPi = zeros(dlogpi_r,dlogpi_c);
    %     df = 1;
    
    for step = 1 : max(size(data(trial).u))-1
        %         G=policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(step));
        %         disp(G(1))
        evalg = eval(subs(g, [w; s; a], [wnum; data(trial).x(step,:)'; data(trial).u(step,:)']));
%         disp(evalg')
        %         if isnan(evalg)
        %             evalg
        %         end
        sumdlogPi = sumdlogPi + evalg;
        
        %         disp('sumdlogPi')
        %         disp(sumdlogPi)
        sumrew = sumrew + gamma^(step-1) * data(trial).r(step,obj);
        
        %         disp('sumrew')
        %         disp(sumrew)
        %         sumrew = sumrew + df * data(trial).r(step);
        
        %         df = df * gamma;
    end
%     disp('---');
%     disp(sumdlogPi'-cppdata.histGradient(trial).g)
%     disp(sumrew-cppdata.J(trial))
    dJdtheta = dJdtheta + sumdlogPi * sumrew;
end

if gamma == 1
    dJdtheta = dJdtheta / j;
else
    %     dJdtheta = (1 - gamma) * dJdtheta / num_trials;
    dJdtheta = dJdtheta / num_trials;
end
