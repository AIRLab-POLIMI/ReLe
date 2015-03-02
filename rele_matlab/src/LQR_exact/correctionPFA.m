%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Saves and plots the solutions found by PFA with and without correction 
% step. After an optimization step, the script saves the current solution
% in 'notcorrectedJ', then perform the necessary correction steps to go
% back on the frontier and saves the corrected solution in 'correctedJ'.
% He then continues the exploration ignoring the correction step.
%
% See 'frontPFA' for PFA more details.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

N = 2;
LQR = init_lqr(N);
g = LQR.g;
A = LQR.A;
B = LQR.B;
Q = LQR.Q;
R = LQR.R;
e = LQR.e;
x0 = LQR.x0;
Sigma = LQR.Sigma;

initK = zeros(N);
for i = 1 : N
%     K(i,i) = unifrnd(-1,0);
    initK(i,i) = -0.5;
end

tolerance_step = 0.1;
tolerance_corr = 0.1;
lrate_single = 0.1;
lrate_step = 0.1;
lrate_corr = 0.1;
total_iter = 0;
normalize_step = true;
normalize_correction = true;


%% Learn the last objective
while true

    total_iter = total_iter + 1;
    nat_grad_init = calcNatGradient(A,B,Q{N},R{N},initK,Sigma,g);
    dev = norm(diag(nat_grad_init));
    if dev < tolerance_step
        break
    end
    initK = initK - lrate_single * nat_grad_init;
    
end

initJ = zeros(1,N);
for i = 1 : N
    initJ(i) = calcJ(A,B,Q{i},R{i},initK,Sigma,x0,g);
end
notcorrectedK = {initK}; % store solutions before starting the correction step
notcorrectedJ = initJ;
correctedK = {initK}; % store solutions found at the end of the correction step
correctedJ = initJ;


%% Learn the remaining objectives
for obj = 1 : N-1

    current_frontK = notcorrectedK;
    
    for i = 1 : numel(current_frontK)

        current_K = current_frontK{i};
        current_iter = 0;

        while true

            total_iter = total_iter + 1;
            current_iter = current_iter + 1;
    
            nat_grad_step = calcNatGradient(A,B,Q{obj},R{obj},current_K,Sigma,g);
            dev = norm(diag(nat_grad_step));
            if dev < tolerance_step % break if the objective has been learned
                break
            end
            
            lrate_step = 0.25 / current_iter; % high on purpose to move away from the frontier after an optimization step
            current_K = current_K - lrate_step * nat_grad_step; % perform an optimization step
            current_J = zeros(1,N);
            for j = 1 : N
                current_J(j) = calcJ(A,B,Q{j},R{j},current_K,Sigma,x0,g);
            end
            notcorrectedK = [notcorrectedK; current_K]; % save the solution before the correction step
            notcorrectedJ = [notcorrectedJ; current_J];
            
            correction_K = current_K; % use a new variable to not overwrite the current K
            while true % perform a correction step
                
                M = zeros(N^2,N);
                Mn = zeros(N^2,N);
                for j = 1 : N
                    nat_grad_corr = calcNatGradient(A,B,Q{j},R{j},correction_K,Sigma,g);
                    M(:,j) = nat_grad_corr(:);
                    Mn(:,j) = nat_grad_corr(:) / max(1,norm(diag(nat_grad_corr)));
                end
                options = optimset('Display','off');
                lambda = quadprog(M'*M, zeros(N,1), [], [], ones(1,N), ...
                    1, zeros(1,N), [], ones(N,1)/N, options);
                dir = M*lambda;
                dirn = Mn*lambda;
                dev = norm(dir);
                devn = norm(dirn);
                if dev < tolerance_corr
                    correction_J = zeros(1,N);
                    for j = 1 : N
                        correction_J(j) = calcJ(A,B,Q{j},R{j},correction_K,Sigma,x0,g);
                    end
                    correctedK = [correctedK; correction_K]; % save Pareto solution
                    correctedJ = [correctedJ; correction_J];
                    break
                end

                nat_grad_corr = vec2mat(dir,N);
                nat_grad_corr_n = vec2mat(dirn,N);
                correction_K = correction_K - lrate_corr * nat_grad_corr; % move towards the frontier
                total_iter = total_iter + 1;
                
            end
            
        end
        
    end
    
end

toc;

%% Plot
figure; hold all
if N == 2
    plot(notcorrectedJ(:,1),notcorrectedJ(:,2),'g+')
    plot(correctedJ(:,1),correctedJ(:,2),'r+')
    xlabel('J_1'); ylabel('J_2');
    legend('Not corrected','Corrected','Location','NorthOutside')
    title(['\epsilon = ' num2str(e) ', \gamma = ' num2str(g) ...
        ', \Sigma = I, x_0 = ' mat2str(x0)])
end

if N == 3
    scatter3(notcorrectedJ(:,1),notcorrectedJ(:,2),notcorrectedJ(:,3),'g+')
    scatter3(correctedJ(:,1),correctedJ(:,2),correctedJ(:,3),'r+')
    xlabel('J_1'); ylabel('J_2'); zlabel('J_3');
    legend('Not corrected','Corrected','Location','NorthOutside')
    title(['\epsilon = ' num2str(e) ', \gamma = ' num2str(g) ...
        ', \Sigma = I, x_0 = ' mat2str(x0)])
end
