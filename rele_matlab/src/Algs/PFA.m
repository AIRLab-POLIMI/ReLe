%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finds the Pareto-frontier using the Pareto-Following Algorithm and the 
% Natural Gradient.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

verboseOut = true; % to print output messages

domain = 'dam';
[N_obj, init_pol, episodes, steps, gamma, avg_rew_setting, max_obj] = settings(domain);
N_params = length(init_pol.theta);

tolerance_step = 0.01; % tolerance on the norm of the gradient (stopping condition) during the optimization step
tolerance_corr = 0.1; % the same, but during the correction step
minH = -20; % min entropy of the policy (with Gaussian policies the (differential) entropy can be negative)
normalize_step = true; % normalize gradients during an optimization step?
lrate_single = 4; % lrate during single-objective optimization phase
lrate_step = 4; % lrate during optimization step
lrate_corr = 4; % lrate during correction step

front_J = []; % Pareto-frontier solutions
front_pol = [];
inter_J = []; % intermediate solutions (during correction phase)
inter_pol = [];

n_iterations = 0; % total number of iterations (policy evaluations)

%% Learn the last objective
while true

    n_iterations = n_iterations + 1;
    [ds, uJ, dJ, H] = collect_samples(domain,episodes,steps,init_pol,avg_rew_setting,gamma);
    nat_grad = eNAC(init_pol,ds,gamma,N_obj);
    dev = norm(nat_grad);
    if dev < tolerance_step || H < minH
        break
    end
    if normalize_step
        nat_grad = nat_grad / max(norm(nat_grad),0);
    end
    init_pol = init_pol.update(lrate_single * nat_grad);
    
end

% save the first solution, i.e. one extreme point of the frontier
if gamma == 1
    J_init = (uJ .* max_obj)';
else
    J_init = (dJ .* max_obj)';
end
front_pol = [front_pol; init_pol];
front_J = [front_J; J_init];

str = strtrim(sprintf('%g, ', J_init));
str(end) = [];
if verboseOut, fprintf('Initial Pareto solution found: [ %s ], H: %g \n', str, H); end

%% Learn the remaining objectives
for obj = N_obj-1 : 1 % for all the remaining objectives ...

    current_front_pol = front_pol;
    
    %%% RANDOMIZATION: SEE README!
    current_front_pol = current_front_pol.randomize;
    
    num_policy = numel(current_front_pol);
    
    for i = 1 : num_policy % ... for all the Pareto-optimal solutions found so far ...

        current_pol = current_front_pol(i);
        current_iter = 0; % number of steps for the single-objective optimization

        if verboseOut, fprintf('\n\nOptimizing objective %d ...\n', obj); end
        
        while true % ... perform policy gradient optimization

            n_iterations = n_iterations + 1;
            current_iter = current_iter + 1;
    
            [ds, uJ, dJ, H] = collect_samples(domain,episodes,steps,current_pol,avg_rew_setting,gamma);
            nat_grad_step = eNAC(current_pol,ds,gamma,obj);
            dev = norm(nat_grad_step);
            
            if verboseOut, fprintf('%d / %d ) ... moving ... Norm: %g, H: %g \n', i, num_policy, dev, H); end
            
            if dev < tolerance_step || H < minH % stopping conditions
                if gamma == 1
                    current_J = (uJ .* max_obj)';
                else
                    current_J = (dJ .* max_obj)';
                end
                front_pol = [front_pol; current_pol]; % save Pareto solution
                front_J = [front_J; current_J];
                
                str = strtrim(sprintf('%g, ', current_J));
                str(end) = [];
                if verboseOut, fprintf('Objective %d optimized! [ %s ], H: %g \n-------------\n', obj, str, H); end
                
                break
            end
            
            if normalize_step
                nat_grad_step = nat_grad_step / max(norm(nat_grad_step),0);
            end
            current_pol = current_pol.update(lrate_step * nat_grad_step); % perform an optimization step
            
            while true % correction phase
                
                [ds, uJ, dJ, H] = collect_samples(domain,episodes,steps,current_pol,avg_rew_setting,gamma);
                if gamma == 1
                    current_J = (uJ .* max_obj)';
                else
                    current_J = (dJ .* max_obj)';
                end
                
                M = zeros(N_params,N_obj);
                for j = 1 : N_obj
                    nat_grad = eNAC(current_pol,ds,gamma,j);
                    M(:,j) = nat_grad / norm(nat_grad); % always normalize during correction
                end
                options = optimset('Display','off');
                lambda = quadprog(M'*M, zeros(N_obj,1), [], [], ones(1,N_obj), ...
                    1, zeros(1,N_obj), [], ones(N_obj,1)/N_obj, options);
                pareto_dir = M*lambda; % minimal-norm Pareto-ascent direction
                dev = norm(pareto_dir);
                
                str = strtrim(sprintf('%g, ', current_J));
                str(end) = [];
                if verboseOut, fprintf('... correcting ... Norm: %g, H: %g, J: [ %s ]\n', dev, H, str); end
                
                if dev < tolerance_corr % if on the frontier
                    front_pol = [front_pol; current_pol]; % save Pareto solution
                    front_J = [front_J; current_J];
                    break
                end
                
                inter_pol = [inter_pol; current_pol]; % save intermediate solution
                inter_J = [inter_J; current_J];

                if H < minH % deterministic policy not on the frontier
                    break
                end

                current_pol = current_pol.update(lrate_corr * pareto_dir); % move towards the frontier

                n_iterations = n_iterations + 1;
                
            end
            
        end
        
    end
    
end

toc;

%% Plot
figure; hold all
if N_obj == 2
    plot(front_J(:,1),front_J(:,2),'g+')
    xlabel('J_1'); ylabel('J_2');
end

if N_obj == 3
    scatter3(front_J(:,1),front_J(:,2),front_J(:,3),'g+')
    xlabel('J_1'); ylabel('J_2'); zlabel('J_3');
end
