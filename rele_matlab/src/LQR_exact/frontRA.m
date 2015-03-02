%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finds the Pareto-frontier using the Radial Algorithm and the Natural 
% Gradient.
% The policy is Gaussian with linear mean and constant variance.
% Only the mean is learned.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

N_obj = 3;
step = 100; % density of the directions in the simplex
tolerance = 0.01; % tolerance for the norm of the gradient
maxLevel = 200; % max number of policy gradient steps in the same direction
lrate = 0.5;

LQR = init_lqr(N_obj);
g = LQR.g;
A = LQR.A;
B = LQR.B;
Q = LQR.Q;
R = LQR.R;
x0 = LQR.x0;
e = LQR.e;
Sigma = LQR.Sigma;

% generate all combinations of weights for the directions in the simplex
W = generate_convex_weights(N_obj, step);
N_sol = size(W,1);
FrontJ = zeros(N_sol, N_obj); % Pareto-frontier solutions
FrontK = cell(N_sol,1);
inter_J = []; % intermediate solutions
inter_K = [];

% initial policy
initK = zeros(N_obj);
for i = 1 : N_obj
    initK(i,i) = -0.5;
end

% initial solution
initJ = zeros(1,N_obj);
initM = zeros(N_obj^2,N_obj);
for j = 1 : N_obj
    initJ(j) = calcJ(A,B,Q{j},R{j},initK,Sigma,x0,g);
    nat_grad = calcNatGradient(A,B,Q{j},R{j},initK,Sigma,g);
    initM(:,j) = nat_grad(:);
end

% for all the directions in the simplex
n_iterations = 0;
parfor k = 1 : N_sol
    
    fixedLambda = W(k,:)';
    newDir = initM * fixedLambda;
    K = initK - lrate * reshape(newDir,N_obj,N_obj);

    level = 1;

    % start a policy gradient learning in that direction
    while true
        
        n_iterations = n_iterations + 1;

        J = zeros(1,N_obj);
        M = zeros(N_obj^2,N_obj);
        for j = 1 : N_obj
            J(j) = calcJ(A,B,Q{j},R{j},K,Sigma,x0,g);
            nat_grad = calcNatGradient(A,B,Q{j},R{j},K,Sigma,g);
            M(:,j) = nat_grad(:);
        end

        options = optimset('Display','off');
        lambdaPareto = quadprog(M'*M, zeros(N_obj,1), [], [], ones(1,N_obj), ...
            1, zeros(1,N_obj), [], ones(N_obj,1)/N_obj, options);
        dirPareto = M * lambdaPareto;
        devPareto = norm(dirPareto);
        
        dir = M * fixedLambda;
        dev = norm(dir);

        if dev < tolerance
            FrontJ(k,:) = J;
            FrontK{k} = K;
            break;
        end

        if devPareto < tolerance
            FrontJ(k,:) = J;
            FrontK{k} = K;
            break;
        end
        
        if level > maxLevel
            FrontJ(k,:) = J;
            FrontK{k} = K;
            break;
        end
        
%         inter_J = [inter_J; J'];
%         inter_K = [inter_K; K];
        level = level + 1;

        K = K - lrate * reshape(dir,N_obj,N_obj);

    end
    
end

toc;

%% Plot
figure; hold all
if N_obj == 2
    plot(FrontJ(:,1),FrontJ(:,2),'g+')
    plot(initJ(:,1),initJ(:,2),'k*','DisplayName','Starting point')
    xlabel('J_1'); ylabel('J_2');
    legend(['\epsilon = ' num2str(e) ', \gamma = ' num2str(g) ...
        ', \Sigma = I, x_0 = ' mat2str(x0)],'Location','NorthOutside')
end

if N_obj == 3
    scatter3(FrontJ(:,1),FrontJ(:,2),FrontJ(:,3),'g+')
    scatter3(initJ(:,1),initJ(:,2),initJ(:,3),'k*','DisplayName','Starting point');
    xlabel('J_1'); ylabel('J_2'); zlabel('J_3');
    legend(['\epsilon = ' num2str(e) ', \gamma = ' num2str(g) ...
        ', \Sigma = I, x_0 = ' mat2str(x0)],'location','NorthOutside')
end
