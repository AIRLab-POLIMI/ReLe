%%% Gaussian with constant mean and logistic covariance.
%%% Both mean and variance are learned.
classdef constant_logistic_gaussian_policy
    
    properties(GetAccess = 'public', SetAccess = 'private')
        % These properties are public by default
        dim;
        tau;
        dim_variance_params;
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        theta;
    end
    
    methods
        
        function obj = constant_logistic_gaussian_policy(dim, ... 
                init_mean, init_sigma_w, max_variance)
            % Class constructor
            assert(size(max_variance,1) == size(init_sigma_w,1));
            assert(size(max_variance,2) == size(init_sigma_w,2));
            obj.theta = [init_mean; init_sigma_w];
            obj.dim   = dim;
            obj.tau   = max_variance;
            obj.dim_variance_params = length(init_sigma_w);
        end
        
        function probability = evaluate(obj, action)
            assert(size(action, 2) == 1);
            assert(size(action, 1) == obj.dim);
            
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            SIGMA = diag(logv);
            MU    = obj.theta(1:obj.dim);

            probability = mvnpdf(action, MU, SIGMA);
        end
        
        function action = drawAction(obj)
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            SIGMA = diag(logv);
            MU    = obj.theta(1:obj.dim);

            action = mvnrnd(MU, SIGMA)';
        end
        
        function dlpdt = dlogPidtheta(obj, action)
            if (nargin == 1)
                % Return the dimension of the vector theta
                dlpdt = size(obj.theta,1);
                return
            end
            
            assert(size(action, 2) == 1);
            assert(size(action, 1) == obj.dim);
            
            % Compute covariance matrix
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            MU    = obj.theta(1:obj.dim);
            
            dlpdt = zeros(size(obj.theta));
            
            dmu = (action - MU) ./ logv;
            dlpdt(1:end-obj.dim) = dmu(:);
            
            for i = 1 : obj.dim
                wi = obj.theta(end-obj.dim+i);
                A = -0.5 * exp(-wi) / (1 + exp(-wi));
                B = 0.5 * exp(-wi) / obj.tau(i) * (action(i) - MU(i))^2;
                dlpdt(end-obj.dim+i) = A + B;
            end
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
        end
        
        % differential entropy, can be negative
        function H = entropy(obj, state)
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            SIGMA = diag(logv);
            H = 0.5*log( (2*pi*exp(1))^obj.dim * det(SIGMA) );
        end
        
        function obj = makeDeterministic(obj)
            obj.tau = 1e-8 * ones(size(obj.tau));
        end
        
        function [mu, sigma] = getMuSigma(obj)
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            sigma = diag(logv);
        end
        
        function obj = weightedMLUpdate(obj, d, Theta)
            assert(size(Theta,2) == size(d,1));
            assert(size(d,2) == 1);
            d = d / sum(d);
            mu = Theta * d;
            logv = zeros(obj.dim,1);
            
            for j = 1 : obj.dim
                tmp = 0;
                for k = 1 : size(Theta,2)
                    tmp = tmp + (d(k) * (Theta(j,k) - mu(j)).^2);
                end
                
                logv(j) = -log( obj.tau(j) / tmp - 1 );
            end
                
            obj.theta = [mu; logv];
        end
        
    end
    
end
