classdef CREPS_Solver < handle

    % Contextual Relative Entropy Policy Search

    properties(GetAccess = 'public', SetAccess = 'private')
        epsilon; % KL divergence bound
        N;       % number of rollouts per iteration
        N_eval;  % number of rollouts to evaluate the policy
        N_MAX;   % how many rollouts (including the ones from previous
                 % distributions) will be used for the update step
        policy;  % distribution for sampling the episodes
        basis;   % basis functions for approximating the value function
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = CREPS_Solver(epsilon, N, N_eval, N_MAX, policy, basis)
            obj.epsilon = epsilon;
            obj.N = N;
            obj.N_eval = N_eval;
            obj.N_MAX = N_MAX;
            obj.policy = policy;
            obj.basis = basis;
        end
        
        %% SETTER
        function obj = setPolicy(obj, policy)
            obj.policy = policy;
        end
        
        %% CORE
        function [d, divKL] = optimize(obj, J, PhiVfun)
            dim_phi = obj.basis();
            
            % Optimization problem settings
            options = optimset('Algorithm', 'interior-point', ...
                'GradObj', 'on', ...
                'Display', 'off', ...
                'MaxFunEvals', 10 * 5, ...
                'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 10);
            
            lowerBound_theta = -ones(dim_phi, 1) * 1e8;
            upperBound_theta = ones(dim_phi, 1) * 1e8;
            lowerBound_eta = 1e-8;
            upperBound_eta = 1e8;
            theta = ones(dim_phi,1);
            eta = 1;
            
            maxIter = 100;
            validKL = false;
            validSF = false;
            numStepsNoKL = 0;
            
            % Iteratively solve fmincon for eta and theta separately
            for i = 1 : maxIter
                if ~validKL || numStepsNoKL < 5
                    eta = fmincon(@(eta)obj.dual_eta(eta,theta,J,PhiVfun), ...
                        eta, [], [], [], [], lowerBound_eta, upperBound_eta, [], options);
                    
                    % Numerical trick
                    advantage = J - theta' * PhiVfun';
                    maxAdvantage = max(advantage);
                    
                    % Compute the weights
                    d = exp( (advantage - maxAdvantage) / eta )';
                    
                    % Check conditions
                    qWeighting = ones(obj.N_MAX,1);
                    pWeighting = d;
                    pWeighting = pWeighting / sum(pWeighting);
                    divKL = getKL(pWeighting, qWeighting);
                    error = divKL - obj.epsilon;
                    validKL = error < 0.1 * obj.epsilon;
                    featureDiff = sum(bsxfun(@times, PhiVfun, pWeighting)) - mean(PhiVfun);
                    validSF = max(abs(featureDiff)) < 0.1;
                    numStepsNoKL = numStepsNoKL + 1;
                end
                
                if ~validSF
                    theta = fmincon(@(theta)obj.dual_theta(theta,eta,J,PhiVfun), ...
                        theta, [], [], [], [], lowerBound_theta, upperBound_theta, [], options);
                    
                    % Numerical trick
                    advantage = J - theta' * PhiVfun';
                    maxAdvantage = max(advantage);
                    
                    % Compute the weights
                    d = exp( (advantage - maxAdvantage) / eta )';
                    
                    % Check conditions
                    qWeighting = ones(obj.N_MAX,1);
                    pWeighting = d;
                    pWeighting = pWeighting / sum(pWeighting);
                    divKL = getKL(pWeighting, qWeighting);
                    error = divKL - obj.epsilon;
                    validKL = error < 0.1 * obj.epsilon;
                    featureDiff = sum(bsxfun(@times, PhiVfun, pWeighting)) - mean(PhiVfun);
                    validSF = max(abs(featureDiff)) < 0.1;
                end
                
                if validSF && validKL
                    break
                end
            end
        end
        
        function update (obj, weights, Delta, PhiPolicy)
            obj.policy = obj.policy.weightedMLUpdate(weights, Delta, PhiPolicy);
        end

        
        %% DUAL FUNCTIONS
        function [g, gd] = dual_full(obj, params, J, Phi)
            theta = params(1:end-1);
            eta = params(end);
            
            V = theta' * Phi';
            n = length(J);
            advantage = J - V;
            maxAdvantage = max(advantage);
            weights = exp( ( advantage - maxAdvantage ) / eta ); % numerical trick
            sumWeights = sum(weights);
            sumWeightsV = sum( weights .* (advantage - maxAdvantage) );
            meanFeatures = mean(Phi)';
            sumWeightsPhi = ( weights * Phi )';
            
            % dual function
            g = eta * obj.epsilon + theta' * meanFeatures + eta * log(sumWeights/n) + maxAdvantage;
            % gradient wrt to theta and to eta
            gd = [meanFeatures - sumWeightsPhi / sumWeights;
                obj.epsilon + log(sumWeights/n) - sumWeightsV / (eta * sumWeights)];
        end
        
        function [g, gd, h] = dual_eta(obj, eta, theta, J, Phi)
            V = theta' * Phi';
            n = length(J);
            advantage = J - V;
            maxAdvantage = max(advantage);
            weights = exp( ( advantage - maxAdvantage ) / eta ); % numerical trick
            sumWeights = sum(weights);
            sumWeightsV = sum( weights .* (advantage - maxAdvantage) );
            sumWeightsV2 = sum( weights .* (advantage - maxAdvantage).^2 );
            meanFeatures = mean(Phi)';
            
            % dual function
            g = eta * obj.epsilon + theta' * meanFeatures + eta * log(sumWeights/n) + maxAdvantage;
            % gradient wrt to eta
            gd = obj.epsilon + log(sumWeights/n) - sumWeightsV / (eta * sumWeights);
            % hessian
            h = ( sumWeightsV2 * sumWeights + eta * sumWeightsV * sumWeights + sumWeightsV^2 ) / ( eta^3 * sumWeightsV^2 );
        end
        
        function [g, gd, h] = dual_theta(obj, theta, eta, J, Phi)
            V = theta' * Phi';
            n = length(J);
            advantage = J - V;
            maxAdvantage = max(advantage);
            weights = exp( ( advantage - maxAdvantage ) / eta ); % numerical trick
            sumWeights = sum(weights);
            meanFeatures = mean(Phi)';
            sumWeightsPhi = ( weights * Phi )';
            sumPhiWeights = (Phi' * weights');
            sumPhiWeightsPhi = Phi' * bsxfun( @times, weights', Phi );
            
            % dual function
            g = eta * obj.epsilon + theta' * meanFeatures + eta * log(sumWeights/n) + maxAdvantage;
            % gradient wrt to theta
            gd = meanFeatures - sumWeightsPhi / sumWeights;
            % hessian
            h = ( sumPhiWeightsPhi * sumWeights - sumPhiWeights * sumWeightsPhi') / sumWeights^2 / eta;
        end

    end
    
end
