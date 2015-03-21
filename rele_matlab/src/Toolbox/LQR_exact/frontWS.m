%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finds the Pareto-frontier using the weighted sum approach.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

step = 1000; % the inverse of the stepsize in the weights interval
N_obj = 2;

LQR = init_lqr(N_obj);
g = LQR.g;
A = LQR.A;
B = LQR.B;
x0 = LQR.x0;
Sigma = LQR.Sigma;

% generate all combinations of weights for the objectives
W = generate_convex_weights(N_obj, step);
N_sol = size(W,1);
FrontJ = zeros(N_sol, N_obj);
FrontK = cell(N_sol,1);
FrontSigma = cell(N_sol,1);

for k = 1 : N_sol

    Q = zeros(N_obj);
    R = zeros(N_obj);
    for i = 1 : N_obj
        Q = Q + W(k,i) * LQR.Q{i};
        R = R + W(k,i) * LQR.R{i};
    end
    
    % compute the optimal controller in closed form
    P = eye(size(Q,1));
    for j = 1 : 100
        K = -g*(R+g*(B'*P*B))\B'*P*A;
        P = Q + g*A'*P*A + g*K'*B'*P*A + g*A'*P*B*K + g*K'*B'*P*B*K + K'*R*K;
    end
    K = -g*(R+g*B'*P*B)\B'*P*A;
    
    % evaluate objectives
    J = zeros(1,N_obj);
    for i = 1 : N_obj
        J(i) = calcJ(A,B,LQR.Q{i},LQR.R{i},K,Sigma,x0,g);
    end
    FrontJ(k,:) = J;
    FrontK{k} = K;
    FrontSigma{k} = Sigma;
    
end
    
figure
hold all

toc;

%% Plot
if N_obj == 2
    plot(FrontJ(:,1),FrontJ(:,2),'g+')
    xlabel('J_1'); ylabel('J_2');
    legend(['\epsilon = ' num2str(LQR.e) ', \gamma = ' num2str(LQR.g) ...
        ', \Sigma = I, x_0 = ' mat2str(LQR.x0)],'Location','NorthOutside')
end

if N_obj == 3
    scatter3(FrontJ(:,1),FrontJ(:,2),FrontJ(:,3),'g+')
    xlabel('J_1'); ylabel('J_2'); zlabel('J_3');
    legend(['\epsilon = ' num2str(LQR.e) ', \gamma = ' num2str(LQR.g) ...
        ', \Sigma = I, x_0 = ' mat2str(LQR.x0)],'location','NorthOutside')
end
