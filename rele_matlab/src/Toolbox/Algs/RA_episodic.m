%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finds the Pareto-frontier using the Radial Algorithm and Natural 
% Evolution Strategies.
% 
% Reference: S Parisi, M Pirotta, N Smacchia, L Bascetta, M Restelli (2014)
% Policy gradient approaches for multi-objective sequential decision making
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

clear all

verboseOut = true; % to print output messages

domain = 'deep';
[N_obj, pol_low] = settings(domain);

% If the policy has a learnable variance, we don't want to learn it and
% we make it deterministic (see 'collect_episodes')
dim_theta = size(pol_low.theta,1) - pol_low.dim_variance_params;

mu0 = zeros(dim_theta,1);
sigma0 = 10 * eye(dim_theta); % change according to the domain

% init_pol = constant_smart_gaussian_policy(dim_theta,mu0,sigma0);
% init_pol = constant_chol_gaussian_policy(dim_theta,mu0,chol(sigma0));
init_pol = constant_diag_gaussian_policy(dim_theta,mu0,sqrt(diag(sigma0)));
N_params = size(init_pol.theta,1);

N_episodes = 50;
step = 10; % density of the directions in the simplex
tolerance = 0.1; % tolerance for the norm of the gradient
maxLevel = 200; % max number of policy gradient steps in the same direction
lrate = 2;

% generate all combinations of weights for the directions in the simplex
W = generate_convex_weights(N_obj, step);
N_sol = size(W,1);
front_J = zeros(N_sol, N_obj); % Pareto-frontier solutions
front_pol = cell(N_sol,1);
inter_J = []; % intermediate solutions
inter_pol = [];

% initial solution
[J_init, Theta_init] = collect_episodes(domain, N_episodes, init_pol);
M_init = zeros(N_params,N_obj);
for j = 1 : N_obj
    M_init(:,j) = NESbase(init_pol, J_init(:,j), Theta_init, lrate);
end

J_init = mean(J_init);
str_init = strtrim(sprintf('%.4f, ', J_init));
str_init(end) = [];
if verboseOut, fprintf('Starting from solution: [ %s ] \n', str_init); end

n_iterations = 0; % total number of iterations (policy evaluations)

%% Run the algorithm
% for all the directions in the simplex
parfor k = 1 : N_sol
    
    fixedLambda = W(k,:)';
    newDir = M_init * fixedLambda;
    curr_pol = init_pol.update(lrate * newDir);

    level = 1;
    str_dir = strtrim(sprintf('%.3f, ', fixedLambda));
    str_dir(end) = [];

    % start a policy gradient learning in that direction
    while true
        
        n_iterations = n_iterations + 1;

        [J, Theta] = collect_episodes(domain, N_episodes, curr_pol);

        M = zeros(N_params,N_obj); % jacobian
        Mn = zeros(N_params,N_obj); % jacobian with normalized gradients
        for j = 1 : N_obj
            nat_grad = NESbase(curr_pol, J(:,j), Theta, lrate);
            M(:,j) = nat_grad;
            Mn(:,j) = nat_grad / max(norm(nat_grad), 1e-8); % to avoid numerical problems
        end
        
        % for the min-norm Pareto-ascent direction, always use Mn
        dirPareto = paretoAscentDir(N_obj, Mn);
        devPareto = norm(dirPareto);
        
        dir = M * fixedLambda;
        dev = norm(dir);

        str_obj = strtrim(sprintf('%.4f, ', mean(J)));
        str_obj(end) = [];
        if verboseOut, fprintf('[ %s ] || LV %d ) Norm: %.4f, \t J = [ %s ] \n', ...
            str_dir, level, dev, str_obj); end

        if dev < tolerance
            if verboseOut, fprintf('Cannot proceed any further in this direction!\n'); end
            front_J(k,:) = mean(J);
            front_pol{k} = curr_pol;
            break;
        end

        if devPareto < tolerance
            if verboseOut, fprintf('Pareto front reached!\n'); end
            front_J(k,:) = mean(J);
            front_pol{k} = curr_pol;
            break;
        end
        
        if level > maxLevel
            if verboseOut, fprintf('Iteration limit reached!\n'); end
            front_J(k,:) = mean(J);
            front_pol{k} = curr_pol;
            break;
        end
        
%         inter_J = [inter_J; J];
%         inter_pol = [inter_pol; curr_pol];
        level = level + 1;

        curr_pol = curr_pol.update(dir);

    end
    
end

toc;

%% Plot
figure; hold all
[f, p] = pareto(front_J, front_pol);
if N_obj == 2
    plot(f(:,1),f(:,2),'g+')
    plot(J_init(:,1),J_init(:,2),'k*','DisplayName','Starting point')
    xlabel('J_1'); ylabel('J_2');
end

if N_obj == 3
    scatter3(f(:,1),f(:,2),f(:,3),'g+')
    scatter3(J_init(:,1),J_init(:,2),J_init(:,3),'k*','DisplayName','Starting point');
    xlabel('J_1'); ylabel('J_2'); zlabel('J_3');
end

plot_reference_fronts(domain);