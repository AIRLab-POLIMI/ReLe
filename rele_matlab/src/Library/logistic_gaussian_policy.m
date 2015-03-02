%%% Gaussian with linear mean and logistic covariance.
%%% Both mean and variance are learned.
classdef logistic_gaussian_policy 
    
    properties(GetAccess = 'public', SetAccess = 'private')
        % These properties are public by default
        basis;
        dim;
        dim_variance_params;
        tau;
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        theta;
    end
    
    methods
        
        function obj = logistic_gaussian_policy(basis, dim, init_k, ...
                init_sigma_w, max_variance)
            % Class constructor
            assert(size(max_variance,1) == size(init_sigma_w,1));
            assert(size(max_variance,2) == size(init_sigma_w,2));
            obj.theta = [init_k(:); init_sigma_w];
            obj.basis = basis;
            obj.dim   = dim;
            obj.tau   = max_variance;
            obj.dim_variance_params = length(init_sigma_w);
        end
        
        function probability = evaluate(obj, state, action)
            assert(size(state, 2) == 1);
            assert(size(action, 2) == 1);
            assert(size(action, 1) == obj.dim);
            
            % Compute covariance matrix
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            SIGMA = diag(logv);
            
            % Compute mean vector
            phi   = feval(obj.basis, state);
            k     = vec2mat(obj.theta(1:end-obj.dim),obj.dim);
            MU    = k * phi;
            probability = mvnpdf(action, MU, SIGMA);
        end
        
        function action = drawAction(obj, state)
            assert(size(state,2) == 1);
            
            % Compute covariance matrix
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            SIGMA = diag(logv);
            
            % Compute mean vector
            phi    = feval(obj.basis, state);
            k      = vec2mat(obj.theta(1:end-obj.dim),obj.dim);
            MU     = k * phi;
            action = mvnrnd(MU, SIGMA)';
        end
        
        function dlpdt = dlogPidtheta(obj, state, action)
            if (nargin == 1)
                % Return the dimension of the vector theta
                dlpdt = size(obj.theta,1);
                return
            end
            
            assert(size(state,  2) == 1);
            assert(size(action, 2) == 1);
            assert(size(action, 1) == obj.dim);
            
            % Compute covariance matrix
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            SIGMA_INV = diag(1./logv);
            
            % Compute mean vector
            phi   = feval(obj.basis, state);
            k     = vec2mat(obj.theta(1:end-obj.dim),obj.dim);
            MU    = k * phi;
            
            dlpdt = zeros(size(obj.theta));
            
            dmu = SIGMA_INV * (action - MU) * phi';
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
        
        function obj = randomize(obj)
            obj.theta(end-obj.dim+1:end) = 1; % change if necessary
        end
        
        function phi = phi(obj, state)
            if (nargin == 1)
                % Return the dimension of the vector of basis functions
                phi = feval(obj.basis);
                return
            end
            phi = feval(obj.basis, state);
        end
        
    end
    
end
