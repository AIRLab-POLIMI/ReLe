%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: D Wierstra, T Schaul, T Glasmachers, Y Sun, J Peters
% J Schmidhuber (2014)
% Natural Evolution Strategy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef NES_Solver < handle
    
    % Natural Evolutionary Strategies
    
    properties(GetAccess = 'public', SetAccess = 'private')
        lrate;
        N;       % number of rollouts per iteration
        N_MAX;   % how many rollouts (including the ones from previous
        % distributions) will be used for the update step
        policy;  % distribution for sampling the episodes
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = NES_Solver(lrate, N, N_MAX, policy)
            obj.lrate = lrate;
            obj.N = N;
            obj.N_MAX = N_MAX;
            obj.policy = policy;
        end
        
        %% SETTER
        function obj = setPolicy(obj, policy)
            obj.policy = policy;
        end
        
        %% CORE
        function div = step(obj, J, Theta)
            [nat_grad, div] = NESbase(obj, J, Theta);
            update(obj, nat_grad);
        end
        
        function [nat_grad, div] = NESbase(obj, J, Theta)
            n_episodes = length(J);
            
            num = 0;
            den = 0;
            dlogPidtheta = zeros(obj.policy.dlogPidtheta,n_episodes);
            
            % Compute optimal baseline
            for k = 1 : n_episodes
                
                dlogPidtheta(:,k) = obj.policy.dlogPidtheta(Theta(:,k));
                
                num = num + dlogPidtheta(:,k).^2 * J(k);
                den = den + dlogPidtheta(:,k).^2;
                
            end
            
            b = num ./ den;
            b(isnan(b)) = 0;
            % b = mean_J;
            
            % Estimate gradient and Fisher information matrix
            grad = 0;
            F = 0;
            for k = 1 : n_episodes
                grad = grad + dlogPidtheta(:,k) .* (J(k) - b);
                F = F + dlogPidtheta(:,k) * dlogPidtheta(:,k)';
            end
            grad = grad / n_episodes;
            F = F / n_episodes;
            
            % If we can compute the FIM in closed form, use it
            if ismethod(obj.policy,'fisher')
                F = obj.policy.fisher;
            end
            
            % If we can compute the FIM inverse in closed form, use it
            if ismethod(obj.policy,'inverseFisher')
                invF = obj.policy.inverseFisher;
                lambda = sqrt(grad' * (invF * grad) / (4 * obj.lrate));
                lambda = max(lambda,1e-8); % to avoid numerical problems
                nat_grad = invF * grad / (2 * lambda);
            elseif rank(F) == size(F,1)
                lambda = sqrt(grad' * (F \ grad) / (4 * obj.lrate));
                lambda = max(lambda,1e-8);
                nat_grad = F \ grad / (2 * lambda);
            else
                str = sprintf('WARNING: F is lower rank (rank = %d)!!! Should be %d', rank(F), size(F,1));
                disp(str);
                lambda = sqrt(grad' * (pinv(F) * grad) / (4 * obj.lrate));
                lambda = max(lambda,1e-8);
                nat_grad = pinv(F) * grad / (2 * lambda);
            end
            
            div = norm(nat_grad);
        end
        
        function update(obj, gradient)
            obj.policy = obj.policy.update(gradient);
        end
        
    end
    
end
