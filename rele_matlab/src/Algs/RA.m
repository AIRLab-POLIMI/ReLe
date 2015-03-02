%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finds the Pareto-frontier using the Radial Algorithm and the Natural 
% Gradient.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

verboseOut = true; % to print output messages

domain = 'dam';
[N_obj, init_pol, episodes, steps, gamma, avg_rew_setting, max_obj] = settings(domain);
N_params = length(init_pol.theta);

step = 5; % density of the directions in the simplex
tolerance = 0.1; % tolerance for the norm of the gradient
maxLevel = 200; % max number of policy gradient steps in the same direction
minH = 0.1; % min entropy of the policy (with Gaussian policies the (differential) entropy can be negative)
lrate = 2;
normalize_step = true; % normalize gradients during a step?

% generate all combinations of weights for the directions in the simplex
W = generate_convex_weights(N_obj, step);
N_sol = size(W,1);
front_J = zeros(N_sol, N_obj); % Pareto-frontier solutions
front_pol = cell(N_sol,1);
inter_J = []; % intermediate solutions
inter_pol = [];

% initial solution
[ds, uJ, dJ] = collect_samples(domain,episodes,steps,init_pol,avg_rew_setting,gamma);
if gamma == 1
    J_init = (uJ .* max_obj)';
else
    J_init = (dJ .* max_obj)';
end
M_init = zeros(N_params,N_obj);
for j = 1 : N_obj
    nat_grad = eNAC(init_pol,ds,gamma,j);
    M_init(:,j) = nat_grad / max(norm(nat_grad),1);
end

str_init = strtrim(sprintf('%g, ', J_init));
str_init(end) = [];
if verboseOut, fprintf('Starting from solution: [ %s ] \n', str_init); end

n_iterations = 0; % total number of iterations (policy evaluations)

% for all the directions in the simplex
parfor k = 1 : N_sol
    
    fixedLambda = W(k,:)';
    newDir = M_init * fixedLambda;
    curr_pol = init_pol.update(lrate * newDir);

    level = 1;
    str_dir = strtrim(sprintf('%g, ', fixedLambda));
    str_dir(end) = [];

    % start a policy gradient learning in that direction
    while true
        
        n_iterations = n_iterations + 1;

        [ds, uJ, dJ, H] = collect_samples(domain,episodes,steps,curr_pol,avg_rew_setting,gamma);
        if gamma == 1
            J = (uJ .* max_obj)';
        else
            J = (dJ .* max_obj)';
        end
        M = zeros(N_params,N_obj); % jacobian
        Mn = zeros(N_params,N_obj); % jacobian with normalized gradients
        for j = 1 : N_obj
            nat_grad = eNAC(curr_pol,ds,gamma,j);
            M(:,j) = nat_grad;
            Mn(:,j) = nat_grad / norm(nat_grad);
        end

        % for the min-norm Pareto-ascent direction, always use Mn
        options = optimset('Display','off');
        lambdaPareto = quadprog(Mn'*Mn, zeros(N_obj,1), [], [], ones(1,N_obj), ...
            1, zeros(1,N_obj), [], ones(N_obj,1)/N_obj, options);
        dirPareto = Mn * lambdaPareto;
        devPareto = norm(dirPareto);
        
        if normalize_step
            dir = Mn * fixedLambda;
        else
            dir = M * fixedLambda;
        end
        dev = norm(dir);

        str_obj = strtrim(sprintf('%g, ', J));
        str_obj(end) = [];
        if verboseOut, fprintf('[ %s ] || LV %g )  \t Norm: %g, \t J = [ %s ] \n', ...
            str_dir, level, dev, str_obj); end

        if dev < tolerance
            if verboseOut, fprintf('Cannot proceed any further in this direction!\n'); end
            front_J(k,:) = J;
            front_pol{k} = curr_pol;
            break;
        end

        if devPareto < tolerance
            if verboseOut, fprintf('Pareto front reached!\n'); end
            front_J(k,:) = J;
            front_pol{k} = curr_pol;
            break;
        end
        
        if H < minH
            if verboseOut, fprintf('Deterministic policy found!\n'); end
            front_J(k,:) = J;
            front_pol{k} = curr_pol;
            break;
        end
        
        if level > maxLevel
            if verboseOut, fprintf('Iteration limit reached!\n'); end
            front_J(k,:) = J;
            front_pol{k} = curr_pol;
            break;
        end
        
%         inter_J = [inter_J; J];
%         inter_pol = [inter_pol; curr_pol];
        level = level + 1;

        curr_pol = curr_pol.update(lrate * dir);

    end
    
end

toc;

%% Plot
figure; hold all
if N_obj == 2
    plot(front_J(:,1),front_J(:,2),'g+')
    plot(J_init(:,1),J_init(:,2),'k*','DisplayName','Starting point')
    xlabel('J_1'); ylabel('J_2');
end

if N_obj == 3
    scatter3(front_J(:,1),front_J(:,2),front_J(:,3),'g+')
    scatter3(J_init(:,1),J_init(:,2),J_init(:,3),'k*','DisplayName','Starting point');
    xlabel('J_1'); ylabel('J_2'); zlabel('J_3');
end
