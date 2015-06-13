clear all
domain = 'dam';
makeDet = 0; % 1 to consider only deterministic policies

[~, pol_low] = settings(domain);
[n_obj, n_params, mu0, sigma0] = settings_episodic(domain,makeDet);
[~, ~, utopia, antiutopia] = getReferenceFront(domain,0);

dist = gaussian_constant(n_params,mu0,sigma0);
% dist = gaussian_chol_constant(n_params,mu0,chol(sigma0));
% dist = gaussian_diag_constant(n_params,mu0,sqrt(diag(sigma0)));

%%% Mixture Model
n_gauss = 5;
mu = zeros(n_gauss,n_params);
sigma = zeros(n_params,n_params,n_gauss);
for i = 1 : n_gauss
    mu(i,:) = dist.drawAction;
    sigma(:,:,i) = sigma0;
end
p = ones(n_gauss,1) / n_gauss;
pol_high = gmm_constant(mu,sigma,p,n_gauss);

% %%% Simple Gaussian
% pol_high = dist;

% fitnessfunc = @(varargin)-eval_loss(varargin{:},domain);
% fitnessfunc = @(varargin)hypervolume(varargin{:},antiutopia,utopia,1e6);
fitnessfunc = @(varargin)hypervolume2d(varargin{:},antiutopia,utopia);

N = 50;
N_MAX = N*10;
epsilon = 0.5;
solver = REPS_Solver(epsilon,N,N_MAX,pol_high);

J = zeros(N_MAX,n_obj);
Theta = zeros(n_params,N_MAX);

iter = 0;

%% Learning
close all
fig = figure;
frames = getframe(fig);
while true
    
    iter = iter + 1;
    
    Theta_iter = zeros(n_params,N);
    pol_iter = pol_low.empty(N,0);
    
    % Draw N policies and evaluate them
    for i = 1 : N
        Theta_iter(:,i) = solver.policy.drawAction;
        pol_iter(i) = pol_low;
        pol_iter(i).theta(1:n_params) = Theta_iter(:,i);
    end
    J_iter = evaluate_policies ( pol_iter, domain, makeDet );
    
    % At first run, fill the pool to maintain the samples distribution
    if iter == 1
        J = repmat(min(J_iter),N_MAX,1);
        for k = 1 : N_MAX
            Theta(:,k) = solver.policy.drawAction;
        end
    end
        
    % Enqueue the new samples and remove the old ones
    J = [J_iter; J(1:N_MAX-N,:)];
    Theta = [Theta_iter, Theta(:, 1:N_MAX-N)];
    
%     % NSGA-II fitness: rank a policy according to the numer of solutions
%     % by which it is dominated
%     C = mat2cell(J,ones(N_MAX,1),n_obj);
%     dominance = cellfun( ...
%         @(X) sum( sum( bsxfun(@ge, J, X), 2) == n_obj ), ...
%         C, 'UniformOutput', false);
%     fitness = - vertcat(dominance{:}) + 1;
    
    % SMS-EMOA fitness: rank a policy according to its contribution to the
    % hypervolume of the frontier
    [uniqueJ, ~, idx] = unique(J,'rows');
    fitnessUnique = zeros(size(uniqueJ,1),1);
    parfor i = 1 : size(uniqueJ,1)
        front_tmp = uniqueJ;
        front_tmp(i,:) = [];
        fitnessUnique(i) = - fitnessfunc(pareto(front_tmp));
    end
    fitness = fitnessUnique(idx);
    
    [weights, divKL] = solver.optimize(fitness);

    avgRew = fitnessfunc(pareto(J));
    fprintf( 'Iter: %d, Avg Reward: %.4f, KL Div: %.2f, Entropy: %.4f\n', ...
        iter, avgRew, divKL, solver.policy.entropy );
    
%%%%%%%%%%%%%%%%%%%%%%%%%%
%     clf
%     subplot(1,2,1,'align')
%     plot(J(:,1),J(:,2),'g+')
%     getReferenceFront(domain,1);
%     title(num2str(length(solver.policy.p)))
% 
%     subplot(1,2,2,'align')
%     plot(J(:,1),J(:,2),'g+')
%     getReferenceFront(domain,1);
%     axis([-2.5 -0.3 -11 -9.2])
%     drawnow
%%%%%%%%%%%%%%%%%%%%%%%%%%

frames = moviefront(fig, frames, domain, J);
    
    
    if divKL < 1e-2
        break
    else
        solver.update(weights, Theta);
    end
    
end

%% Eval
Theta_eval = zeros(n_params,N_MAX);
pol_eval = pol_low.empty(N_MAX,0);

for i = 1 : N_MAX
    Theta_eval(:,i) = solver.policy.drawAction;
    pol_eval(i) = pol_low;
    pol_eval(i).theta(1:n_params) = Theta_eval(:,i);
end

f_eval = evaluate_policies ( pol_eval, domain, makeDet );

%% Plot
%%%%%%%%%%%%%%
ref = getReferenceFront(domain,1);
idx = f_eval(:,1) < max(ref(:,1));
f_eval = f_eval(idx,:);

idx = f_eval(:,2) < max(ref(:,2));
f_eval = f_eval(idx,:);
%%%%%%%%%%%%%%
[f, p] = pareto(f_eval, pol_eval);

figure; hold all
if n_obj == 2
    plot(f(:,1),f(:,2),'g+')
    xlabel('J_1'); ylabel('J_2');
end

if n_obj == 3
    scatter3(f(:,1),f(:,2),f(:,3),'g+')
    xlabel('J_1'); ylabel('J_2'); zlabel('J_3');
end

getReferenceFront(domain,1);

fprintf('Fitness: %.4f\n', fitnessfunc(f));
