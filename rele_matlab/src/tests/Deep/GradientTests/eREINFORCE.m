function dJdtheta = eREINFORCE(policy, data, gamma, obj, cppdata)

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
        evalg = policyg.dlogPidtheta(data(trial).x(step,:)', 1+data(trial).u(step,:)');
        %        evalg = eval(subs(g, [w; s; a], [wnum; data(trial).x(step,:)'; data(trial).u(step,:)']));
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
    dJdtheta = dJdtheta / num_trials;
else
    %     dJdtheta = (1 - gamma) * dJdtheta / num_trials;
    dJdtheta = dJdtheta / num_trials;
end
