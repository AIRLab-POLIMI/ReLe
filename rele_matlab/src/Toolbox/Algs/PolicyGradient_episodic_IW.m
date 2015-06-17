% clear all

domain = 'deep';
robj = 1;
[n_obj, n_params, mu0, sigma0] = settings_episodic(domain,1);

% pol_high = gaussian_constant(n_params,mu0,sigma0);
% pol_high = gaussian_chol_constant(n_params,mu0,chol(sigma0));
pol_high = gaussian_diag_constant(n_params,mu0,sqrt(diag(sigma0)));

N = 10;
N_MAX = N * 10;
N_eval = 100;
lrate = 0.1;

J = zeros(N_MAX,n_obj);
Theta = zeros(pol_high.dim,N_MAX);
Policies = pol_high.empty(N_MAX,0);

J_history = [];
iter = 0;

%% Learning
while iter < 200

    iter = iter + 1;
    
    
    
    
    J_eval = collect_episodes(domain, N_eval, pol_high);
    
    

    
    [J_iter, Theta_iter] = collect_episodes(domain, N, pol_high);
    Policies_iter(1:N) = pol_high;
    
    % At first run, fill the pool to maintain the samples distribution
    if iter == 1
        J = repmat(min(J_iter),N_MAX,1);
        for k = 1 : N_MAX
            Theta(:,k) = pol_high.drawAction;
            Policies(k) = pol_high;
        end
    end
    
    % Enqueue the new samples and remove the old ones
    J = [J_iter; J(1:N_MAX-N,:)];
    Theta = [Theta_iter, Theta(:, 1:N_MAX-N)];
    Policies = [Policies_iter, Policies(1:N_MAX-N)];

    % Importance Sampling Weights
    W = zeros(N_MAX, 1); % IS weights
    p = zeros(N_MAX, 1); % p(i) = probability of drawing Theta_i from policy p (p = target)
    Q = zeros(N_MAX, N_MAX); % Q(i,j) = probability of drawing Theta_i from policy q_j (q = sampling)
    alpha = ones(1, N_MAX) / N_MAX; % mixture responsibilities
    for i = 1 : N_MAX
        p(i) = pol_high.evaluate(Theta(:,i));
        for j = 1 : N : N_MAX
            Q(i, j:j+N-1) = Policies(j).evaluate(Theta(:,i));
        end
%         W(i) = p(i) / Q(i,i); % Standard IW
        W(i) = p(i) / sum(alpha .* Q(i,:)); % Mixture IW
%         W(i) = p(i) / sum(Q(i,:)); % Daniel, IROS 2012
%         W(i) = 1;
    end
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % Diagnostic
%     omega = zeros(N_MAX, 1);
%     for k = 1 : N : N_MAX
%         omega(k) = Q(k,k) / sum(Q(k,:));
%     end
%     
%     % importance sampling estimate
%     muq_hat = sum(J(:,robj).*W) / N_MAX;
%     
%     % variance estimate
%     s2_hat = sum((J(:,robj).*W - muq_hat).^2) / N_MAX;
% 
%     % self-normalized importance sampling estimate
%     muq_tilde = sum(J(:,robj).*W) / sum(W);
%     
%     % variance estimate
%     s2_tilde = sum(W.^2.*(J(:,robj) - muq_tilde).^2) / sum(W);
%     
%     s = 0;
%     for k = 1 : N_MAX
%         s = s + (W(k) - mean(W))^2;
%     end
%     cw = sqrt(1/(N_MAX-1)*s) / mean(W);
%     
%     n = N_MAX / (1 + cw^2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    
    
    [grad, stepsize] = NES_IWbase(pol_high, J(:,robj), Theta, W, lrate);
%     [grad, stepsize] = PGPE_IWbase(pol_high, J(:,robj), Theta, W, lrate);

    avgRew = mean(J_iter(:,robj));
    J_history = [J_history, J_eval(:,robj)];
%     fprintf( 'Iter: %d, Avg Reward: %.4f, Norm: %.2f, Entropy: %.3f\n', ...
%         iter, avgRew, norm(grad), pol_high.entropy );

    % Ending condition
    if norm(grad) < 0.0001
%         break
    else
        pol_high = pol_high.update(grad * stepsize);
    end
    
end