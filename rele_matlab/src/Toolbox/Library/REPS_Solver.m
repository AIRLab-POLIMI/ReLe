classdef REPS_Solver < handle

    % Relative Entropy Policy Search

    properties(GetAccess = 'public', SetAccess = 'private')
        epsilon; % KL divergence bound
        N;       % number of rollouts per iteration
        N_MAX;   % how many rollouts (including the ones from previous
                 % distributions) will be used for the update step
        policy;  % distribution for sampling the episodes
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = REPS_Solver(epsilon, N, N_MAX, policy)
            obj.epsilon = epsilon;
            obj.N = N;
            obj.N_MAX = N_MAX;
            obj.policy = policy;
        end
        
        %% SETTER
        function obj = setPolicy(obj, policy)
            obj.policy = policy;
        end
        
        %% CORE
        function div = step(obj, J, Delta)
            [d, div] = optimize(obj, J);
            update(obj, d, Delta);
        end
        
        function [d, divKL] = optimize(obj, J)
            % Optimization problem settings
            options = optimset('GradObj', 'on', ...
                'Display', 'off', ...
                'MaxFunEvals', 300 * 5, ...
                'Algorithm', 'interior-point', ...
                'TolX', 10^-8, ...
                'TolFun', 10^-12, ...
                'MaxIter', 300);
            lowerBound = 1e-8; % eta > 0
            upperBound = 1e8; % eta < inf
            eta0 = 1;
            eta = fmincon(@(eta)obj.dual(eta,J), ...
                eta0, [], [], [], [], lowerBound, upperBound, [], options);
            
            % Perform weighted ML to update the high-level policy
            d = exp( (J - max(J)) / eta );

            % Compute KL divergence
            qWeighting = ones(obj.N_MAX,1);
            pWeighting = d;
            divKL = getKL(pWeighting, qWeighting);
        end
        
        function update (obj, weights, Delta)
            obj.policy = obj.policy.weightedMLUpdate(weights, Delta);
        end

        
        %% DUAL FUNCTION
        function [g, gd] = dual(obj, eta, J)
            % Numerical trick
            maxJ = max(J);
            J = J - maxJ;
            
            A = sum(exp(J / eta)) / obj.N_MAX;
            B = sum(exp(J / eta) .* J) / obj.N_MAX;
            
            g = eta * obj.epsilon + eta * log(A) + maxJ; % dual function
            gd = obj.epsilon + log(A) - B / (eta * A);   % gradient
        end

    end
    
end
