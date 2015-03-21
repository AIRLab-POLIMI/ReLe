%%% Gaussian with constant mean and covariance.
%%% Both mean and variance are learned.
classdef constant_smart_gaussian_policy
    
    properties(GetAccess = 'public', SetAccess = 'private')
        dim;
        dim_variance_params;
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        theta
    end
    
    methods
        
        function obj = constant_smart_gaussian_policy(dim, init_mean, init_sigma)
            assert(isscalar(dim))
            assert(size(init_mean,1) == dim)
            assert(size(init_mean,2) == 1)
            assert(size(init_sigma,1) == dim)
            assert(size(init_sigma,2) == dim)

            obj.theta = [init_mean; init_sigma(:)];
            obj.dim = dim;
            obj.dim_variance_params = length(init_sigma(:));
        end
        
        function param = drawAction(obj)
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            sigma = vec2mat(obj.theta(obj.dim+1:end),obj.dim);
            param = mvnrnd(mu,sigma)';
        end
        
        % differential entropy, can be negative
        function H = entropy(obj, state)
            sigma = vec2mat(obj.theta(obj.dim+1:end),obj.dim);
            H = 0.5*log( (2*pi*exp(1))^obj.dim * det(sigma) );
        end
        
        function probability = evaluate(obj, theta)
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            sigma = vec2mat(obj.theta(obj.dim+1:end),obj.dim);
            probability = mvnpdf(theta, mu, sigma);
        end
        
        function dlogpdt = dlogPidtheta(obj, theta)
            if (nargin == 1)
                % Return the dimension of the gradient
                dlogpdt = length(obj.theta);
                return
            end
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            sigma = vec2mat(obj.theta(obj.dim+1:end),obj.dim);
            
            dlogpdt_mu = sigma \ (theta - mu);
            A = -.5 * eye(obj.dim) / sigma;
            B = .5 * sigma \ (theta - mu) * (theta - mu)' / sigma;
            dlogpdt_sigma = A + B;
            
            dlogpdt = [dlogpdt_mu; dlogpdt_sigma(:)];
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
            sigma = vec2mat(obj.theta(obj.dim+1:end),obj.dim);
            obj.theta(obj.dim+1:end) = nearestSPD(sigma);
        end

        function obj = makeDeterministic(obj)
            obj.theta(obj.dim+1:end) = 1e-8;
        end
        
        function params = getParams(obj)
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            sigma = vec2mat(obj.theta(obj.dim+1:end),obj.dim);
            
            params.mu = mu;
            params.Sigma = sigma;
        end
       
        function obj = weightedMLUpdate(obj, d, Theta)
            mu = Theta * d / sum(d);
            sigma = zeros(size(obj.dim));
            for k = 1 : size(Theta,2)
                sigma = sigma + (d(k) * (Theta(:,k) - mu) * (Theta(:,k) - mu)');
            end
            Z = (sum(d)^2 - sum(d.^2)) / sum(d);
            sigma = sigma / Z;
            sigma = nearestSPD(sigma);
            obj.theta = [mu; sigma(:)];
        end
        
        function obj = randomize(obj, factor)
            obj.theta(obj.dim+1:end) = obj.theta(obj.dim+1:end) * factor;
        end
       
    end
    
end
