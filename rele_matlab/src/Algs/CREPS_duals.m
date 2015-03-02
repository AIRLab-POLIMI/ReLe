classdef CREPS_duals
    
    properties
    end
    
    methods(Static)
        
        function [g, gd] = full(params, J, epsilon, Phi)
            
            theta = params(1:end-1);
            eta = params(end);
            
            V = theta' * Phi';
            N = length(J);
            advantage = J - V;
            maxAdvantage = max(advantage);
            weights = exp( ( advantage - maxAdvantage ) / eta ); % numerical trick
            sumWeights = sum(weights);
            sumWeightsV = sum( weights .* (advantage - maxAdvantage) );
            meanFeatures = mean(Phi)';
            sumWeightsPhi = ( weights * Phi )';
            
            % dual function
            g = eta * epsilon + theta' * meanFeatures + eta * log(sumWeights/N) + maxAdvantage;
            % gradient wrt to theta and to eta
            gd = [meanFeatures - sumWeightsPhi / sumWeights;
                epsilon + log(sumWeights/N) - sumWeightsV / (eta * sumWeights)];
            
        end
        
        function [g, gd, h] = eta(eta, theta, J, epsilon, Phi)
            
            V = theta' * Phi';
            N = length(J);
            advantage = J - V;
            maxAdvantage = max(advantage);
            weights = exp( ( advantage - maxAdvantage ) / eta ); % numerical trick
            sumWeights = sum(weights);
            sumWeightsV = sum( weights .* (advantage - maxAdvantage) );
            sumWeightsV2 = sum( weights .* (advantage - maxAdvantage).^2 );
            meanFeatures = mean(Phi)';
            
            % dual function
            g = eta * epsilon + theta' * meanFeatures + eta * log(sumWeights/N) + maxAdvantage;
            % gradient wrt to eta
            gd = epsilon + log(sumWeights/N) - sumWeightsV / (eta * sumWeights);
            % hessian
            h = ( sumWeightsV2 * sumWeights + eta * sumWeightsV * sumWeights + sumWeightsV^2 ) / ( eta^3 * sumWeightsV^2 );
            
        end
        
        function [g, gd, h] = theta(theta, eta, J, epsilon, Phi)
            
            V = theta' * Phi';
            N = length(J);
            advantage = J - V;
            maxAdvantage = max(advantage);
            weights = exp( ( advantage - maxAdvantage ) / eta ); % numerical trick
            sumWeights = sum(weights);
            meanFeatures = mean(Phi)';
            sumWeightsPhi = ( weights * Phi )';
            sumPhiWeights = (Phi' * weights');
            sumPhiWeightsPhi = Phi' * bsxfun( @times, weights', Phi );
            
            % dual function
            g = eta * epsilon + theta' * meanFeatures + eta * log(sumWeights/N) + maxAdvantage;
            % gradient wrt to theta
            gd = meanFeatures - sumWeightsPhi / sumWeights;
            % hessian
            h = ( sumPhiWeightsPhi * sumWeights - sumPhiWeights * sumWeightsPhi') / sumWeights^2 / eta;
            
        end
        
    end
    
end
