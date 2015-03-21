%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finds the Pareto-frontier using the Pareto-Following Algorithm and 
% Natural Evolution Strategies.
% 
% Reference: Parisi, S.; Pirotta, M.; Smacchia, N.; Bascetta, L.; Restelli, M.
% Policy gradient approaches for multi-objective sequential decision making
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

tic

verboseOut = true; % to print output messages

domain = 'deep';
[N_obj, pol_low] = settings(domain);

% If the policy has a learnable variance, we don't want to learn it and
% we make it deterministic (see 'collect_episodes')
dim_theta = size(pol_low.theta,1) - pol_low.dim_variance_params;

mu0 = zeros(dim_theta,1);
sigma0 = 10 * eye(dim_theta); % change according to the domain

% init_pol = constant_smart_gaussian_policy(n_params,mu0,sigma0);
% init_pol = constant_chol_gaussian_policy(n_params,mu0,chol(sigma0));
init_pol = constant_diag_gaussian_policy(dim_theta,mu0,sqrt(diag(sigma0)));
N_params = size(init_pol.theta,1);

N = 50;
N_MAX = N*10;
tolerance_step = 0.1; % tolerance on the norm of the gradient (stopping condition) during the optimization step
tolerance_corr = 0.1; % the same, but during the correction step
normalize_step = false; % normalize gradients during an optimization step?
lrate_single = 2; % lrate during single-objective optimization phase
lrate_step = 1; % lrate during optimization step
lrate_corr = 2; % lrate during correction step

J = zeros(N_MAX,N_obj);
Theta = zeros(dim_theta,N_MAX);

front_J = []; % Pareto-frontier solutions
front_pol = [];
inter_J = []; % intermediate solutions (during correction phase)
inter_pol = [];

n_iterations = 0; % total number of iterations (policy evaluations)

%% Learn the last objective
while true

    n_iterations = n_iterations + 1;
    [J_init, Theta_init] = collect_episodes(domain, N, init_pol);
    
    % At first run, fill the pool to maintain the samples distribution
    if n_iterations == 1
        J = repmat(min(J_init),N_MAX,1);
        for k = 1 : N_MAX
            Theta(:,k) = init_pol.drawAction;
        end
        J_init_pool = J;
        Theta_init_pool = Theta;
    end
    
    % Enqueue the new samples and remove the old ones
    J = [J_init; J(1:N_MAX-N,:)];
    Theta = [Theta_init, Theta(:, 1:N_MAX-N)];

    nat_grad = NESbase(init_pol, J(:,N_obj), Theta, lrate_single);
    dev = norm(nat_grad);
    if dev < tolerance_step
        break
    end
    if normalize_step
        nat_grad = nat_grad / max(norm(nat_grad),1e-8); % to avoid problems when norm = 0
    end
    init_pol = init_pol.update(nat_grad);
    
end

avgRew = mean(J_init);
% save the first solution, i.e. one extreme point of the frontier
front_pol = [front_pol; init_pol];
front_J = [front_J; avgRew];

str = strtrim(sprintf('%.4f, ', avgRew));
str(end) = [];
if verboseOut, fprintf('Initial Pareto solution found: [ %s ]\n', str); end

%% Learn the remaining objectives
for obj = N_obj-1 : 1 % for all the remaining objectives ...

    current_front_pol = front_pol;

    num_policy = numel(current_front_pol);
    
    for i = 1 : num_policy % ... for all the Pareto-optimal solutions found so far ...

        current_pol = current_front_pol(i);
        current_pol = current_pol.randomize(4); % SEE README!
        current_iter = 0; % number of steps for the single-objective optimization

        % Reset the pool
        J = J_init_pool;
        Theta = Theta_init_pool;

        for k = 1 : N_MAX
            Theta(:,k) = current_pol.drawAction;
        end
        
        if verboseOut, fprintf('\n\nOptimizing objective %d ...\n', obj); end
        
        while true % ... perform policy gradient optimization

            n_iterations = n_iterations + 1;
            current_iter = current_iter + 1;
    
            [J_step, Theta_step] = collect_episodes(domain, N, current_pol);
            
            J = [J_step; J(1:N_MAX-N,:)];
            Theta = [Theta_step, Theta(:, 1:N_MAX-N)];
            avgRew = mean(J_step);
            nat_grad_step = NESbase(current_pol, J(:,obj), Theta, lrate_step);
            dev = norm(nat_grad_step);
            
            if verboseOut, fprintf('%d / %d ) ... moving ... Norm: %.4f \n', i, num_policy, dev); end
            
            if dev < tolerance_step % stopping conditions
                front_pol = [front_pol; current_pol]; % save Pareto solution
                front_J = [front_J; avgRew];
                
                str = strtrim(sprintf('%g, ', avgRew));
                str(end) = [];
                if verboseOut, fprintf('Objective %d optimized! [ %s ] \n-------------\n', obj, str); end
                
                break
            end
            
            if normalize_step
                nat_grad_step = nat_grad_step / max(norm(nat_grad_step),0);
            end
            current_pol = current_pol.update(nat_grad_step); % perform an optimization step
            iter_correction = 0;
            
            while true % correction phase
                
                [J_step, Theta_step] = collect_episodes(domain, N, current_pol);
                J = [J_step; J(1:N_MAX-N,:)];
                Theta = [Theta_step, Theta(:, 1:N_MAX-N)];
                avgRew = mean(J_step);

                M = zeros(N_params,N_obj);
                for j = 1 : N_obj
                    nat_grad = NESbase(current_pol, J(:,j), Theta, lrate_corr);
                    M(:,j) = nat_grad / max(norm(nat_grad),1e-8); % always normalize during correction
                end
                pareto_dir = paretoAscentDir(N_obj, M); % minimal-norm Pareto-ascent direction
                dev = norm(pareto_dir);
                
                str = strtrim(sprintf('%.4f, ', avgRew));
                str(end) = [];
                if verboseOut, fprintf('... correcting ... Norm: %.4f, J: [ %s ]\n', dev, str); end
                
                if dev < tolerance_corr % if on the frontier
                    front_pol = [front_pol; current_pol]; % save Pareto solution
                    front_J = [front_J; avgRew];
                    break
                end
                
                inter_pol = [inter_pol; current_pol]; % save intermediate solution
                inter_J = [inter_J; avgRew];

                current_pol = current_pol.update(pareto_dir); % move towards the frontier

                n_iterations = n_iterations + 1;
                iter_correction = iter_correction + 1;
                
            end
            
        end
        
    end
    
end

toc

%% Plot
figure; hold all
f = pareto(front_J, front_pol);
f_i = pareto(inter_J, inter_pol);
if N_obj == 2
    plot(f(:,1),f(:,2),'g+')
    plot(f_i(:,1),f_i(:,2),'r+')
    xlabel('J_1'); ylabel('J_2');
end

if N_obj == 3
    scatter3(f(:,1),f(:,2),f(:,3),'g+')
    scatter3(f_i(:,1),f_i(:,2),f_i(:,3),'r+')
    xlabel('J_1'); ylabel('J_2'); zlabel('J_3');
end
